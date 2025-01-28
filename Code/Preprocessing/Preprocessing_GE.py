"""
Forecast Data Preprocessing Script

This script preprocesses forecast data for the Iori basin. It resamples data to weekly resolution,
calculates weighted means, interpolates forecast data, and combines everything into a single dataset.

Usage:
1. Ensure all required packages are installed.
2. Update the base directory and file paths as needed.
3. Run the script.

Output:
The combined dataset is saved as a NetCDF file.

"""
from dependencies import (
    # Core data science libraries
    os,
    np,
    xr,
    pd,
    gpd,
    
    # Geometry and interpolation
    box,
    UnivariateSpline,
    
    # Visualization
    plt
)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

base_dir = os.path.join(script_dir, "Data", "Iori")
t2m_path = os.path.join(base_dir, "Hourly_t2m_reanalysis_data_GE.nc")
tp_path = os.path.join(base_dir, "Hourly_tp_reanalysis_data_GE.nc")
forecast_path = os.path.join(base_dir, "Monthly_Forecast_GE.nc")
shapefile_path = os.path.join(base_dir, "Upstream_basin_Iori.shp")
discharge_path = os.path.join(base_dir, "Discharge_GE.nc")

# Load datasets
t2m_ds = xr.open_dataset(t2m_path)
tp_ds = xr.open_dataset(tp_path)
forecast_ds = xr.open_dataset(forecast_path)
study_area = gpd.read_file(shapefile_path)
discharge_ds = xr.open_dataset(discharge_path)


# Ensure CRS is EPSG:4326
study_area = study_area.set_crs(epsg=4326)

# Resample to weekly resolution starting on Monday
weekly_t2m = t2m_ds.t2m.resample(valid_time='1W-MON').mean()
weekly_tp = tp_ds.tp.resample(valid_time='1W-MON').sum()

# Combine resampled variables into a new dataset
weekly_ds = xr.Dataset({'t2m': weekly_t2m, 'tp': weekly_tp})

# Create GeoDataFrame for tiles
lon_res, lat_res = 0.25, 0.25
tiles = [
    {'geometry': box(lon - lon_res / 2, lat - lat_res / 2, lon + lon_res / 2, lat + lat_res / 2), 'latitude': lat, 'longitude': lon}
    for lat in weekly_ds.latitude.values for lon in weekly_ds.longitude.values
]
tiles_gdf = gpd.GeoDataFrame(tiles, crs='EPSG:4326')

# Reproject tiles and filter by study area
tiles_gdf = tiles_gdf.to_crs(study_area.crs)
filtered_tiles = gpd.overlay(tiles_gdf, study_area, how='intersection')

# Calculate area of intersection for each tile
filtered_tiles = filtered_tiles.to_crs(epsg=3857)
filtered_tiles['area'] = filtered_tiles.geometry.area
filtered_tiles = filtered_tiles.to_crs(epsg=4326)

# Calculate weighted means for each week
weighted_means_tp, weighted_means_t2m = [], []
for week in weekly_ds.valid_time:
    weighted_sum_tp, weighted_sum_t2m, total_area = 0, 0, 0
    for _, row in filtered_tiles.iterrows():
        lat, lon, area = row['latitude'], row['longitude'], row['area']
        value_tp = weekly_ds['tp'].sel(valid_time=week, latitude=lat, longitude=lon).item()
        value_t2m = weekly_ds['t2m'].sel(valid_time=week, latitude=lat, longitude=lon).item()
        weighted_sum_tp += value_tp * area
        weighted_sum_t2m += value_t2m * area
        total_area += area
    weighted_means_tp.append(weighted_sum_tp / total_area)
    weighted_means_t2m.append(weighted_sum_t2m / total_area)

# Convert lists to DataArrays and save as NetCDF
weighted_means_tp_da = xr.DataArray(weighted_means_tp, coords=[weekly_ds.valid_time], dims=["valid_time"])
weighted_means_t2m_da = xr.DataArray(weighted_means_t2m, coords=[weekly_ds.valid_time], dims=["valid_time"])
weighted_ds = xr.Dataset({'weighted_tp': weighted_means_tp_da, 'weighted_t2m': weighted_means_t2m_da})
weighted_ds
# Calculate the ensemble mean for t2m and tprate
ensemble_mean_t2m = forecast_ds.t2m.mean(dim='number')
ensemble_mean_tprate = forecast_ds.tprate.mean(dim='number')

# Function for spline interpolation
def spline_interpolation(dates, values, new_dates):
    ordinal_dates = pd.to_datetime(dates).map(pd.Timestamp.toordinal)
    new_ordinal_dates = pd.to_datetime(new_dates).map(pd.Timestamp.toordinal)
    spline = UnivariateSpline(ordinal_dates, values, s=0)
    interpolated_values = spline(new_ordinal_dates)
    return interpolated_values

# Extract the time series for the ensemble mean
forecast_time = pd.to_datetime(forecast_ds.forecast_reference_time.values)
ensemble_mean_t2m_values = ensemble_mean_t2m.sel(latitude=42.0, longitude=44.8).values
ensemble_mean_tprate_values = ensemble_mean_tprate.sel(latitude=42.0, longitude=44.8).values

# Find the overlapping time period
start_date = max(forecast_time.min(), weekly_ds.valid_time.min().values)
end_date = min(forecast_time.max(), weekly_ds.valid_time.max().values)

# Filter the dates to the overlapping period
overlapping_forecast_time = forecast_time[(forecast_time >= start_date) & (forecast_time <= end_date)]
overlapping_weekly_dates = weekly_ds.valid_time[(weekly_ds.valid_time >= start_date) & (weekly_ds.valid_time <= end_date)]

# Initialize lists to store the interpolated values for each forecast month
interpolated_t2m = []
interpolated_tprate = []

# Loop through each forecast month and perform the interpolation
for forecast_month in range(1, 7):
    # Extract the corresponding values for the overlapping period for the current forecast month
    overlapping_ensemble_mean_t2m_values = ensemble_mean_t2m.sel(latitude=42.0, longitude=44.8, forecastMonth=forecast_month).sel(forecast_reference_time=overlapping_forecast_time).values
    overlapping_ensemble_mean_tprate_values = ensemble_mean_tprate.sel(latitude=42.0, longitude=44.8, forecastMonth=forecast_month).sel(forecast_reference_time=overlapping_forecast_time).values

    # Interpolate temperature and precipitation rate to weekly values for the overlapping period
    weekly_forecast_t2m = spline_interpolation(overlapping_forecast_time, overlapping_ensemble_mean_t2m_values, overlapping_weekly_dates)
    weekly_forecast_tprate = spline_interpolation(overlapping_forecast_time, overlapping_ensemble_mean_tprate_values, overlapping_weekly_dates)

    # Store the interpolated values
    interpolated_t2m.append(weekly_forecast_t2m)
    interpolated_tprate.append(weekly_forecast_tprate)

# Create DataArrays for the interpolated forecast data
weekly_forecast_t2m_da = xr.DataArray(interpolated_t2m, coords=[range(1, 7), overlapping_weekly_dates], dims=["forecastMonth", "valid_time"])
weekly_forecast_tprate_da = xr.DataArray(interpolated_tprate, coords=[range(1, 7), overlapping_weekly_dates], dims=["forecastMonth", "valid_time"])

# Shift the valid_time coordinate for each forecastMonth
shifted_forecast_t2m = []
shifted_forecast_tprate = []

for forecast_month in range(1, 7):
    shift_weeks = (forecast_month - 1) * 4
    shifted_time = overlapping_weekly_dates + pd.to_timedelta(shift_weeks, unit='W')
    shifted_forecast_t2m.append(weekly_forecast_t2m_da.sel(forecastMonth=forecast_month).assign_coords(valid_time=shifted_time))
    shifted_forecast_tprate.append(weekly_forecast_tprate_da.sel(forecastMonth=forecast_month).assign_coords(valid_time=shifted_time))

# Combine the shifted forecast data into DataArrays
shifted_forecast_t2m_da = xr.concat(shifted_forecast_t2m, dim='forecastMonth')
shifted_forecast_tprate_da = xr.concat(shifted_forecast_tprate, dim='forecastMonth')

# Create a new time coordinate that spans from the earliest to the latest timestamp
earliest_time = min(weighted_ds.valid_time.min().values, discharge_ds.time.min().values, forecast_time.min())
latest_time = max(weighted_ds.valid_time.max().values, discharge_ds.time.max().values, forecast_time.max())
new_time = pd.date_range(start=earliest_time, end=latest_time, freq='W-MON')

# Reindex the weighted dataset to the new time coordinate, filling missing values with NaN
weighted_ds_reindexed = weighted_ds.reindex(valid_time=new_time, method=None)

# Adding the discharge data
discharge_ds = discharge_ds.where(discharge_ds != -9999, np.nan)
discharge_ds_interpolated = discharge_ds.interpolate_na(dim='time', method='linear')  # primitive but fine since there are almost no NaN values
weekly_discharge = discharge_ds_interpolated.discharge.resample(time='1W-MON').mean()
weekly_discharge_reindexed = weekly_discharge.reindex(time=new_time, method=None).rename({'time': 'valid_time'})

# Convert forecast_tprate from m/s to mm/w
forecast_tprate_converted = shifted_forecast_tprate_da * 1000 * 604800

# Convert weekly_tp from meters to millimeters
weekly_tp_mm = weighted_ds_reindexed.weighted_tp * 1000

# Combine with the existing weighted dataset
combined_ds = xr.Dataset({
    'weighted_tp': weekly_tp_mm,
    'weighted_t2m': weighted_ds_reindexed.weighted_t2m,
    'forecast_t2m': shifted_forecast_t2m_da.reindex(valid_time=new_time, method=None),
    'forecast_tprate': forecast_tprate_converted.reindex(valid_time=new_time, method=None),
    'discharge': weekly_discharge_reindexed
})

combined_ds = combined_ds.drop_vars('number').rename({'forecastMonth': 'lead_time'})


# Save the combined dataset to a NetCDF file
output_path = os.path.join(script_dir, 'Preprocessed_data_GE.nc')
combined_ds.to_netcdf(output_path)


print(f"Combined dataset saved to NetCDF file at {output_path}.")

# Plotting function
def plot_combined_ds(combined_ds):
    variables = ['weighted_t2m', 'weighted_tp', 'forecast_t2m', 'forecast_tprate', 'discharge']
    y_labels = [
        'Temperature in K',
        'Precipitation in mm/w',
        'Temperature Forecast in K',
        'Precipitation Forecast in mm/w',
        'Mean discharge in m$^3$/w'
    ]
    num_vars = len(variables)
    forecast_months = combined_ds.lead_time.values
    fontsize = 10

    fig, axes = plt.subplots(num_vars, 1, figsize=(15, 5 * num_vars), sharex=True)

    for i, var in enumerate(variables):
        ax = axes[i]
        if 'lead_time' in combined_ds[var].dims:
            for j, fm in enumerate(forecast_months):
                combined_ds[var].sel(lead_time=fm).plot(ax=ax, label=f'Lead time {j + 1} months')
        else:
            combined_ds[var].plot(ax=ax, label=var)
        
        ax.set_title('')
        ax.set_ylabel(y_labels[i], fontsize=fontsize)
        ax.grid(True)

        ax.set_xlabel('')

        if i in [0, 1, 4]:
            ax.legend().remove()
        else:
            ax.legend(fontsize=fontsize, loc='upper left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    plt.show()

plot_combined_ds(combined_ds)