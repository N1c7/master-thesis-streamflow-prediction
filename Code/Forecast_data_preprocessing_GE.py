import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Local paths to the files
t2m_path = r"../Data/Iori/Hourly_t2m_reanalysis_data_GE.nc"
tp_path = r"../Data/Iori/Hourly_tp_reanalysis_data_GE.nc"
forecast_path = r"../Data/Iori/Monthly_Forecast_GE.nc"
shapefile_path = r"../Data/Iori/Upstream_basin_Iori.shp"
discharge_path = r"../Data/Iori/Discharge_GE.nc"

# Load the datasets
t2m_ds = xr.open_dataset(t2m_path, engine='netcdf4')
tp_ds = xr.open_dataset(tp_path, engine='netcdf4')
forecast_ds = xr.open_dataset(forecast_path, engine='netcdf4')
study_area = gpd.read_file(shapefile_path)
discharge_ds = xr.open_dataset(discharge_path, engine='netcdf4')

# Resample to weekly resolution
weekly_t2m = t2m_ds.t2m.resample(time='1W').mean()
weekly_tp = tp_ds.tp.resample(time='1W').sum()

# Convert precipitation from meters to millimeters
weekly_tp = weekly_tp * 1000

# Combine the resampled variables into a new dataset
weekly_ds = xr.Dataset({'t2m': weekly_t2m, 'tp': weekly_tp})

# Create a GeoDataFrame for the tiles
lon_res, lat_res = 0.25, 0.25
tiles = [box(lon - lon_res / 2, lat - lat_res / 2, lon + lon_res / 2, lat + lat_res / 2)
         for lat in weekly_ds.latitude.values for lon in weekly_ds.longitude.values]
tiles_gdf = gpd.GeoDataFrame({'geometry': tiles, 'latitude': weekly_ds.latitude.values.repeat(len(weekly_ds.longitude)),
                              'longitude': weekly_ds.longitude.values.tile(len(weekly_ds.latitude))}, crs='EPSG:4326')

# Reproject and filter tiles that intersect with the study area
tiles_gdf = tiles_gdf.to_crs(study_area.crs)
filtered_tiles = gpd.overlay(tiles_gdf, study_area, how='intersection').to_crs(epsg=3857)
filtered_tiles['area'] = filtered_tiles.geometry.area
filtered_tiles = filtered_tiles.to_crs(epsg=4326)

# Calculate the weighted mean for each week
weighted_means_tp, weighted_means_t2m = [], []
for week in weekly_ds.time:
    weighted_sum_tp, weighted_sum_t2m, total_area = 0, 0, 0
    for _, row in filtered_tiles.iterrows():
        lat, lon, area = row['latitude'], row['longitude'], row['area']
        value_tp = weekly_ds['tp'].sel(time=week, latitude=lat, longitude=lon).item()
        value_t2m = weekly_ds['t2m'].sel(time=week, latitude=lat, longitude=lon).item()
        weighted_sum_tp += value_tp * area
        weighted_sum_t2m += value_t2m * area
        total_area += area
    weighted_means_tp.append(weighted_sum_tp / total_area)
    weighted_means_t2m.append(weighted_sum_t2m / total_area)

# Save the weighted means to NetCDF and Excel files
weighted_means_tp_da = xr.DataArray(weighted_means_tp, coords=[weekly_ds.time], dims=["time"])
weighted_means_t2m_da = xr.DataArray(weighted_means_t2m, coords=[weekly_ds.time], dims=["time"])
weighted_means_tp_da.to_netcdf('weighted_means_tp_weekly.nc')
weighted_means_t2m_da.to_netcdf('weighted_means_t2m_weekly.nc')
combined_df = pd.DataFrame({'Date': weighted_means_tp_da.time.values, 'Precipitation (mm)': weighted_means_tp_da.values,
                            'Temperature (K)': weighted_means_t2m_da.values})
combined_df.to_excel('weighted_weekly_climate_data.xlsx', index=False)

# Downscale forecast data
ensemble_mean = forecast_ds.sel(lead_time=1).mean(dim='lead_time')
forecast_time = pd.to_datetime(forecast_ds.date.values)
ensemble_mean_t2m = ensemble_mean.t2m.sel(latitude=42.0, longitude=44.8, method='nearest').values
ensemble_mean_tprate = ensemble_mean.tp.sel(latitude=42.0, longitude=44.8, method='nearest').values

# Create a DataFrame for the forecast data
forecast_df = pd.DataFrame({'Date': forecast_time, 'Ensemble Mean T2M': ensemble_mean_t2m,
                            'Ensemble Mean Precipitation Rate (m/s)': ensemble_mean_tprate})

# Perform spline interpolation for downscaling
def spline_interpolation(dates, values, new_dates):
    ordinal_dates = pd.to_datetime(dates).map(pd.Timestamp.toordinal)
    new_ordinal_dates = pd.to_datetime(new_dates).map(pd.Timestamp.toordinal)
    spline = UnivariateSpline(ordinal_dates, values, s=0)
    return spline(new_ordinal_dates)

weekly_dates = pd.date_range(start='2003-02-24', end='2024-11-25', freq='W-MON')
weekly_t2m = spline_interpolation(forecast_df['Date'], forecast_df['Ensemble Mean T2M'], weekly_dates)
weekly_tprate = spline_interpolation(forecast_df['Date'], forecast_df['Ensemble Mean Precipitation Rate (m/s)'], weekly_dates)
seconds_per_week = 7 * 24 * 60 * 60
weekly_precip = weekly_tprate * seconds_per_week * 1000  # Convert to mm

# Save the downscaled weekly forecast data to a DataFrame
weekly_forecast_df = pd.DataFrame({'Date': weekly_dates, 'Weekly T2M': weekly_t2m, 'Weekly Precipitation (mm)': weekly_precip})
weekly_forecast_df.to_excel('downscaled_weekly_forecast.xlsx', index=False)

# Extract the outflow data
outflow = discharge_ds['discharge'].to_series()

# Combine the data into a single DataFrame
combined_df = pd.DataFrame({
    'Date': weekly_tp.time.values,
    'Precipitation (mm)': weekly_tp.values,
    'Temperature (K)': weekly_t2m.values,
    'Outflow': outflow.reindex(weekly_tp.time.values).values
})

# Save the combined DataFrame to an Excel file
output_excel_path = r"../Data/Iori/weekly_climate_data.xlsx"
combined_df.to_excel(output_excel_path, index=False)

# Optionally, print the combined DataFrame to verify
print(combined_df.head())

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Plot Total Precipitation
axs[0].plot(weekly_tp.time, weekly_tp, label='Total Precipitation (tp) in millimeters')
axs[0].set_title('Time Series of Weekly Aggregated Total Precipitation in mm')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Total Precipitation (tp)')
axs[0].legend()
axs[0].grid(True)

# Plot 2 Metre Temperature
axs[1].plot(weekly_t2m.time, weekly_t2m - 273.15, label='2 Metre Temperature (t2m) in °C')  # Convert from Kelvin to Celsius
axs[1].set_title('Time Series of Weekly Aggregated 2 Metre Temperature in °C')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('2 Metre Temperature (t2m)')
axs[1].legend()
axs[1].grid(True)

# Plot Outflow
axs[2].plot(outflow.index, outflow, label='Outflow')
axs[2].set_title('Time Series of Outflow')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Outflow')
axs[2].legend()
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()