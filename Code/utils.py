
from dependencies import (
    # Core data science libraries
    os,
    np,
    xr,
    pd,
    joblib,
    
    # Statistical metrics and functions
    sklearn_mse,  # mean_squared_error alias
    mae,          # mean_absolute_error alias
    pearsonr,
    
    # Sklearn components
    TimeSeriesSplit,
    StandardScaler
)

def add_lagged_features_split(df, target_column, lags):
    lagged_df = df.copy()
    for lag in lags:
        lagged_df[f'{target_column}_lag_{lag}'] = lagged_df[target_column].shift(lag)
    return lagged_df.dropna()

def preprocessing_data(input_file_path, model_save_dir, lags=[1, 2, 3], n_splits=3):
    # Load the data into an xarray Dataset
    ds = xr.open_dataset(input_file_path)

    # Select input features and output feature
    input_features = ds[['weighted_t2m', 'weighted_tp']].to_dataframe().dropna()
    output_feature = ds['discharge'].to_dataframe().dropna()

    # Align the input and output features
    input_features = input_features.loc[output_feature.index]
    output_feature = output_feature.loc[input_features.index]

    # Split data
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(input_features))
    train_val_idx, test_idx = splits[-1]
    train_idx, val_idx = splits[-2]

    # Create separate training, validation, and test sets with lagged features
    # Training set
    train_features = input_features.iloc[train_idx].copy()
    train_output = output_feature.iloc[train_idx]
    train_features['discharge'] = train_output
    train_features = add_lagged_features_split(train_features, 'discharge', lags)
    y_train = train_features['discharge']
    X_train = train_features.drop(columns=['discharge'])

    # Validation set
    val_features = input_features.iloc[val_idx].copy()
    val_output = output_feature.iloc[val_idx]
    val_features['discharge'] = val_output
    val_features = add_lagged_features_split(val_features, 'discharge', lags)
    y_val = val_features['discharge']
    X_val = val_features.drop(columns=['discharge'])

    # Test set
    test_features = input_features.iloc[test_idx].copy()
    test_output = output_feature.iloc[test_idx]
    test_features['discharge'] = test_output
    test_features = add_lagged_features_split(test_features, 'discharge', lags)
    y_test = test_features['discharge']
    X_test = test_features.drop(columns=['discharge'])

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit only on training data
    X_val = scaler.transform(X_val)          # Apply the same scaler
    X_test = scaler.transform(X_test)        # Apply the same scaler

    output_scaler = StandardScaler()
    y_train = output_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()  # Fit only on training targets
    y_val = output_scaler.transform(y_val.values.reshape(-1, 1)).flatten()          # Apply the same scaler
    y_test = output_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    # Save test data and scaler
    os.makedirs(model_save_dir, exist_ok=True)
    np.save(os.path.join(model_save_dir, 'X_test.npy'), X_test)
    joblib.dump(output_scaler, os.path.join(model_save_dir, 'output_scaler.pkl'))
    np.save(os.path.join(model_save_dir, 'y_test.npy'), y_test)
    test_dates = input_features.index[test_idx]
    np.save(os.path.join(model_save_dir, 'test_dates.npy'), test_dates)

    return X_train, y_train, X_val, y_val, X_test, y_test, ds, output_scaler

def prepare_direct_data(X, y, forecast_horizon):
    X_out, y_out = [], []
    for i in range(len(X) - forecast_horizon):
        X_out.append(X[i:i + forecast_horizon])
        y_out.append(y[i + forecast_horizon])
    return np.array(X_out), np.array(y_out)

def prepare_forecast_data(X, y, forecast_horizon, ds):
    X_out, y_out = [], []
    for i in range(len(X) - forecast_horizon):
        temp_X = X[i:i + forecast_horizon].copy()
        temp_y = y[i + forecast_horizon]
        
        if forecast_horizon == 1:
            X_out.append(temp_X)
            y_out.append(temp_y)
        
        elif 2 <= forecast_horizon <= 4:
            forecast_t2m = ds['forecast_t2m'].sel(lead_time=1).to_dataframe().dropna().values
            forecast_tp = ds['forecast_tprate'].sel(lead_time=1).to_dataframe().dropna().values
            if i + forecast_horizon < len(forecast_t2m) and i + forecast_horizon < len(forecast_tp):
                temp_X[-1, -2:] = [forecast_t2m[i + forecast_horizon][0], forecast_tp[i + forecast_horizon][0]]
            X_out.append(temp_X)
            y_out.append(temp_y)
        
        elif 5 <= forecast_horizon <= 8:
            for j in range(4, forecast_horizon):
                lead_time = 2
                forecast_t2m = ds['forecast_t2m'].sel(lead_time=lead_time).to_dataframe().dropna().values
                forecast_tp = ds['forecast_tprate'].sel(lead_time=lead_time).to_dataframe().dropna().values
                if i + j < len(forecast_t2m) and i + j < len(forecast_tp):
                    temp_X[j, -2:] = [forecast_t2m[i + j][0], forecast_tp[i + j][0]]
            X_out.append(temp_X)
            y_out.append(temp_y)
        
        elif 9 <= forecast_horizon <= 12:
            for j in range(8, forecast_horizon):
                lead_time = 3
                forecast_t2m = ds['forecast_t2m'].sel(lead_time=lead_time).to_dataframe().dropna().values
                forecast_tp = ds['forecast_tprate'].sel(lead_time=lead_time).to_dataframe().dropna().values
                if i + j < len(forecast_t2m) and i + j < len(forecast_tp):
                    temp_X[j, -2:] = [forecast_t2m[i + j][0], forecast_tp[i + j][0]]
            X_out.append(temp_X)
            y_out.append(temp_y)
        
        elif 13 <= forecast_horizon <= 16:
            for j in range(12, forecast_horizon):
                lead_time = 4
                forecast_t2m = ds['forecast_t2m'].sel(lead_time=lead_time).to_dataframe().dropna().values
                forecast_tp = ds['forecast_tprate'].sel(lead_time=lead_time).to_dataframe().dropna().values
                if i + j < len(forecast_t2m) and i + j < len(forecast_tp):
                    temp_X[j, -2:] = [forecast_t2m[i + j][0], forecast_tp[i + j][0]]
            X_out.append(temp_X)
            y_out.append(temp_y)
        
        elif 17 <= forecast_horizon <= 20:
            for j in range(16, forecast_horizon):
                lead_time = 5
                forecast_t2m = ds['forecast_t2m'].sel(lead_time=lead_time).to_dataframe().dropna().values
                forecast_tp = ds['forecast_tprate'].sel(lead_time=lead_time).to_dataframe().dropna().values
                if i + j < len(forecast_t2m) and i + j < len(forecast_tp):
                    temp_X[j, -2:] = [forecast_t2m[i + j][0], forecast_tp[i + j][0]]
            X_out.append(temp_X)
            y_out.append(temp_y)
        
        elif 21 <= forecast_horizon <= 24:
            for j in range(20, forecast_horizon):
                lead_time = 6
                forecast_t2m = ds['forecast_t2m'].sel(lead_time=lead_time).to_dataframe().dropna().values
                forecast_tp = ds['forecast_tprate'].sel(lead_time=lead_time).to_dataframe().dropna().values
                if i + j < len(forecast_t2m) and i + j < len(forecast_tp):
                    temp_X[j, -2:] = [forecast_t2m[i + j][0], forecast_tp[i + j][0]]
            X_out.append(temp_X)
            y_out.append(temp_y)

    return np.array(X_out), np.array(y_out)

def calculate_weights(forecast_horizon):
    """
    Calculate weights based on forecast horizon
    """
    base_weight = 0.7
    decay_rate = 0.1
    forecast_weight = base_weight * (1 - np.exp(-decay_rate * forecast_horizon))
    direct_weight = 1 - forecast_weight
    return direct_weight, forecast_weight

def nash_sutcliffe_efficiency(y_true, y_pred):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE)
    """
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))


def calculate_metrics(y_true, y_pred, label=None):
    """
    Calculate performance metrics
    """
    try:
        rmse = np.sqrt(sklearn_mse(y_true, y_pred))  # Use sklearn_mse here
        mae_value = mae(y_true, y_pred)
        pearson_r, _ = pearsonr(y_true, y_pred)  # Calculate Pearson's r
        nse = nash_sutcliffe_efficiency(y_true, y_pred)  
        mse_value = sklearn_mse(y_true, y_pred)  # Use sklearn_mse here

        if label:
            print(f"\nMetrics for {label}:")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae_value:.4f}")
            print(f"Pearson's r: {pearson_r:.4f}")
            print(f"NSE: {nse:.4f}")
            print(f"MSE: {mse_value:.4f}")
        
        return rmse, mae_value, pearson_r, nse, mse_value
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        print(f"Shape of y_true: {y_true.shape}, Shape of y_pred: {y_pred.shape}")
        return None, None, None, None, None