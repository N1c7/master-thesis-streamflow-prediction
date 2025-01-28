"""
Forecast Data Preprocessing Script

This script preprocesses forecast data for the Secchia basin. It resamples data to weekly resolution,
calculates weighted means, interpolates forecast data, and combines everything into a single dataset.

Usage:
1. Ensure all required packages are installed.
2. Update the base directory and file paths as needed.
3. Run the script.

Output:
The combined dataset is saved as a NetCDF file.

"""

from Code.dependencies import (
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

base_dir = os.path.join(script_dir, "Data", "Secchia")
t2m_path = os.path.join(base_dir, "Hourly_t2m_reanalysis_data_IT.nc")
tp_path = os.path.join(base_dir, "Hourly_tp_reanalysis_data_IT.nc")
forecast_path = os.path.join(base_dir, "Monthly_Forecast_IT.nc")
shapefile_path = os.path.join(base_dir, "Upstream_basin_Secchia.shp")
discharge_path = os.path.join(base_dir, "Discharge_IT.nc")


# Load datasets
t2m_ds = xr.open_dataset(t2m_path)
tp_ds = xr.open_dataset(tp_path)
forecast_ds = xr.open_dataset(forecast_path)
study_area = gpd.read_file(shapefile_path)
discharge_ds = xr.open_dataset(discharge_path)

# Resample to weekly resolution starting on Monday
weekly_t2m = t2m_ds.t2m.resample(valid_time='1W-MON').mean()
weekly_tp = tp_ds.tp.resample(valid_time='1W-MON').sum()
weekly_ds = xr.Dataset({'t2m': weekly_t2m, 'tp': weekly_tp})

# Ensure CRS is EPSG:4326
study_area = study_area.to_crs(epsg=4326)

# Create GeoDataFrame for tiles
lon_res, lat_res = 0.25, 0.25
tiles = [
    {'geometry': box(lon - lon_res / 2, lat - lat_res / 2, lon + lon_res / 2, lat + lat_res / 2), 'latitude': lat, 'longitude': lon}
    for lat in t2m_ds.latitude.values for lon in t2m_ds.longitude.values
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

# Create GeoDataFrame for forecast tiles
lon_res, lat_res = 1.0, 1.0
forecast_tiles = [
    {'geometry': box(lon - lon_res / 2, lat - lat_res / 2, lon + lon_res / 2, lat + lat_res / 2), 'latitude': lat, 'longitude': lon}
    for lat in forecast_ds.latitude.values for lon in forecast_ds.longitude.values
]
forecast_tiles_gdf = gpd.GeoDataFrame(forecast_tiles, crs='EPSG:4326')

# Reproject forecast tiles and filter by study area
forecast_tiles_gdf = forecast_tiles_gdf.to_crs(study_area.crs)
filtered_forecast_tiles = gpd.overlay(forecast_tiles_gdf, study_area, how='intersection')

# Calculate area of intersection for each forecast tile
filtered_forecast_tiles = filtered_forecast_tiles.to_crs(epsg=3857)
filtered_forecast_tiles['area'] = filtered_forecast_tiles.geometry.area
filtered_forecast_tiles = filtered_forecast_tiles.to_crs(epsg=4326)

# Initialize arrays to store the weighted means for each forecast reference time and forecast month
weighted_means_tp = np.full((len(forecast_ds.forecast_reference_time), len(forecast_ds.forecastMonth)), np.nan)
weighted_means_t2m = np.full((len(forecast_ds.forecast_reference_time), len(forecast_ds.forecastMonth)), np.nan)

# Calculate the weighted mean for each forecast reference time and forecast month
for i, ref_time in enumerate(forecast_ds.forecast_reference_time):
    for j, month in enumerate(forecast_ds.forecastMonth):
        weighted_sum_tp = 0
        weighted_sum_t2m = 0
        total_area = 0
        
        for index, row in filtered_forecast_tiles.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            area = row['area']
            
            # Get the value of the variables for the current tile, forecast reference time, and forecast month
            try:
                value_tp = ensemble_mean_tprate.sel(forecast_reference_time=ref_time, forecastMonth=month, latitude=lat, longitude=lon).values
                value_t2m = ensemble_mean_t2m.sel(forecast_reference_time=ref_time, forecastMonth=month, latitude=lat, longitude=lon).values
            except KeyError:
                continue
            
            # Ensure the selection returns a single value
            if value_tp.size == 1 and value_t2m.size == 1:
                value_tp = value_tp.item()
                value_t2m = value_t2m.item()
                
                weighted_sum_tp += value_tp * area
                weighted_sum_t2m += value_t2m * area
                total_area += area
        
        if total_area > 0:
            weighted_mean_tp = weighted_sum_tp / total_area
            weighted_mean_t2m = weighted_sum_t2m / total_area
        else:
            weighted_mean_tp = np.nan
            weighted_mean_t2m = np.nan
        
        weighted_means_tp[i, j] = weighted_mean_tp
        weighted_means_t2m[i, j] = weighted_mean_t2m

# Create a new DataFrame to store the weighted means with the correct dates
dates = []
lead_times = []
tp_values = []
t2m_values = []

for i, ref_time in enumerate(forecast_ds.forecast_reference_time.values):
    for j, month in enumerate(forecast_ds.forecastMonth.values):
        actual_date = pd.Timestamp(ref_time) + pd.DateOffset(months=int(month))
        dates.append(actual_date)
        lead_times.append(month)
        tp_values.append(weighted_means_tp[i, j])
        t2m_values.append(weighted_means_t2m[i, j])

weighted_means_tp_df = pd.DataFrame({'date': dates, 'lead_time': lead_times, 'tp': tp_values})
weighted_means_t2m_df = pd.DataFrame({'date': dates, 'lead_time': lead_times, 't2m': t2m_values})

# Pivot the DataFrames to have lead times as columns
weighted_means_tp_pivot = weighted_means_tp_df.pivot(index='date', columns='lead_time', values='tp')
weighted_means_t2m_pivot = weighted_means_t2m_df.pivot(index='date', columns='lead_time', values='t2m')

# Convert the pivoted DataFrames to DataArrays
forecast_weighted_means_tp_da = xr.DataArray(weighted_means_tp_pivot.values, coords=[weighted_means_tp_pivot.index, weighted_means_tp_pivot.columns], dims=["date", "lead_time"])
forecast_weighted_means_t2m_da = xr.DataArray(weighted_means_t2m_pivot.values, coords=[weighted_means_t2m_pivot.index, weighted_means_t2m_pivot.columns], dims=["date", "lead_time"])
str(forecast_weighted_means_tp_da)

# Function for spline interpolation
def spline_interpolation_with_bounds(dates, values, new_dates):
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 2:  # Ensure there are enough points for interpolation
        return np.full(len(new_dates), np.nan)

    valid_dates = pd.to_datetime(dates[valid_mask])
    valid_values = values[valid_mask]
    
    valid_start = valid_dates.min()
    valid_end = valid_dates.max()
    
    ordinal_dates = valid_dates.map(pd.Timestamp.toordinal)
    new_ordinal_dates = pd.to_datetime(new_dates).map(pd.Timestamp.toordinal)
    
    spline = UnivariateSpline(ordinal_dates, valid_values, s=0)
    interpolated_values = spline(new_ordinal_dates)
    
    out_of_bounds_mask = (new_dates < valid_start) | (new_dates > valid_end)
    interpolated_values[out_of_bounds_mask] = np.nan
    
    return interpolated_values

# Extract the time series for the ensemble mean
forecast_time = pd.to_datetime(forecast_weighted_means_t2m_da.date.values)

# Create a new weekly time coordinate starting from the first available Monday
start_date = forecast_time.min()
end_date = forecast_time.max() + pd.DateOffset(months=6)  # Add 6 months to cover all lead times
weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')

# Initialize lists to store the interpolated values for each lead time
interpolated_t2m = []
interpolated_tp = []

# Loop through each lead time and perform the interpolation
for lead_time in range(1, 7):
    t2m_values = forecast_weighted_means_t2m_da.sel(lead_time=lead_time).values
    tp_values = forecast_weighted_means_tp_da.sel(lead_time=lead_time).values

    weekly_forecast_t2m = spline_interpolation_with_bounds(forecast_time, t2m_values, weekly_dates)
    weekly_forecast_tp = spline_interpolation_with_bounds(forecast_time, tp_values, weekly_dates)

    interpolated_t2m.append(weekly_forecast_t2m)
    interpolated_tp.append(weekly_forecast_tp)

# Create DataArrays for the interpolated forecast data
weekly_forecast_t2m_da = xr.DataArray(
    interpolated_t2m, coords=[range(1, 7), weekly_dates], dims=["lead_time", "valid_time"]
)
weekly_forecast_tp_da = xr.DataArray(
    interpolated_tp, coords=[range(1, 7), weekly_dates], dims=["lead_time", "valid_time"]
)

# Add Discharge
discharge_ds = discharge_ds.where(discharge_ds != -9999, np.nan)
discharge_ds_interpolated = discharge_ds.interpolate_na(dim='valid_time', method='linear')
weekly_discharge = discharge_ds_interpolated.discharge.resample(time='1W-MON').mean()
weekly_discharge_reindexed = weekly_discharge.reindex(time=weekly_dates, method=None).rename({'time': 'valid_time'})

# Convert forecast_tprate from m/s to mm/w
forecast_tprate_converted = weekly_forecast_tp_da * 1000 * 604800

# Convert weekly_tp from meters to millimeters
weekly_tp_mm = weighted_means_tp_da * 1000

# Combine with the existing weighted dataset
combined_ds = xr.Dataset({
    'weighted_tp': weekly_tp_mm.reindex(valid_time=weekly_dates, method=None),
    'weighted_t2m': weighted_means_t2m_da.reindex(valid_time=weekly_dates, method=None),
    'forecast_t2m': weekly_forecast_t2m_da.reindex(valid_time=weekly_dates, method=None),
    'forecast_tprate': forecast_tprate_converted.reindex(valid_time=weekly_dates, method=None),
    'discharge': weekly_discharge_reindexed
})


combined_ds
# Save the combined dataset to a NetCDF file
output_path = os.path.join(script_dir, 'Preprocessed_data_IT.nc')

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
    lead_times = combined_ds.lead_time.values
    fontsize = 10

    fig, axes = plt.subplots(num_vars, 1, figsize=(15, 5 * num_vars), sharex=True)

    for i, var in enumerate(variables):
        ax = axes[i]
        if 'lead_time' in combined_ds[var].dims:
            for j, lt in enumerate(lead_times):
                combined_ds[var].sel(lead_time=lt).plot(ax=ax, label=f'Lead time {j + 1} months')
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

# Plot the combined dataset
plot_combined_ds(combined_ds)