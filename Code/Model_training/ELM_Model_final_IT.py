import os
from project_paths import get_code_dir, get_data_dir, get_models_dir

from dependencies import (
    # Core data science libraries
    np,
    pd,
    plt,
    xr,
    os,
    joblib,
    tf,
    pinv,
    
    # Sklearn components
    TimeSeriesSplit,
    StandardScaler,
    mean_squared_error,
    mean_absolute_error,
    train_test_split,
    
    # Local utility functions
    preprocessing_data,
    prepare_direct_data,
    prepare_forecast_data,
    calculate_weights,
    calculate_metrics
)

np.random.seed(123)
tf.random.set_seed(123)

# Define file paths and model save directory
input_file_path = os.path.join(get_data_dir(), 'Secchia', 'Preprocessed_data_IT.nc')
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
model_save_dir = os.path.join(script_dir, 'Models', 'Secchia', 'ELM')

# Prepare data
X_train, y_train, X_val, y_val, X_test, y_test, ds, output_scaler = preprocessing_data(input_file_path, model_save_dir)


class ELM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_weights = np.random.randn(self.input_size, self.hidden_size)
        self.biases = np.random.randn(self.hidden_size)
        self.output_weights = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        self.output_weights = np.dot(pinv(H), y)

    def predict(self, X):
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)


forecast_horizons = [1, 2, 4, 12, 24]
all_results = {}

for idx, forecast_horizon in enumerate(forecast_horizons, 1):
    print(f"\n{'='*50}")
    print(f"Processing forecast horizon: {forecast_horizon}")
    print(f"{'='*50}")
    
    try:
        # Prepare data for both approaches
        print("Preparing direct data...")
        X_direct_train, y_direct_train = prepare_direct_data(X_train, y_train, forecast_horizon)
        X_direct_val, y_direct_val = prepare_direct_data(X_val, y_val, forecast_horizon)
        X_direct_test, y_direct_test = prepare_direct_data(X_test, y_test, forecast_horizon)

        print("Preparing forecast data...")
        X_forecast_train, y_forecast_train = prepare_forecast_data(X_train, y_train, forecast_horizon, ds)
        X_forecast_val, y_forecast_val = prepare_forecast_data(X_val, y_val, forecast_horizon, ds)
        X_forecast_test, y_forecast_test = prepare_forecast_data(X_test, y_test, forecast_horizon, ds)

        # Flatten the input data for ELM
        input_size = X_direct_train.shape[1] * X_direct_train.shape[2]
        hidden_size = 14  # Number of hidden neurons

        X_direct_train_flat = X_direct_train.reshape(X_direct_train.shape[0], -1)
        X_direct_val_flat = X_direct_val.reshape(X_direct_val.shape[0], -1)
        X_direct_test_flat = X_direct_test.reshape(X_direct_test.shape[0], -1)

        X_forecast_train_flat = X_forecast_train.reshape(X_forecast_train.shape[0], -1)
        X_forecast_val_flat = X_forecast_val.reshape(X_forecast_val.shape[0], -1)
        X_forecast_test_flat = X_forecast_test.reshape(X_forecast_test.shape[0], -1)

        # Initialize and train models
        print("Training direct model...")
        direct_model = ELM(input_size, hidden_size)
        direct_model.fit(X_direct_train_flat, y_direct_train)

        print("Training forecast model...")
        forecast_model = ELM(input_size, hidden_size)
        forecast_model.fit(X_forecast_train_flat, y_forecast_train)

        # Save models using joblib
        direct_model_path = os.path.join(model_save_dir, f'direct_elm_model_FH{forecast_horizon}.joblib')
        forecast_model_path = os.path.join(model_save_dir, f'forecast_elm_model_FH{forecast_horizon}.joblib')
        joblib.dump(direct_model, direct_model_path)
        joblib.dump(forecast_model, forecast_model_path)

        # Generate predictions
        print("Generating predictions...")
        y_pred_direct = direct_model.predict(X_direct_test_flat)
        y_pred_forecast = forecast_model.predict(X_forecast_test_flat)

        # Calculate weights and combine predictions
        direct_weight, forecast_weight = calculate_weights(forecast_horizon)
        y_pred_combined = (direct_weight * y_pred_direct + forecast_weight * y_pred_forecast)

        # Denormalize predictions
        y_test_denorm = output_scaler.inverse_transform(y_direct_test.reshape(-1, 1)).flatten()
        y_pred_direct_denorm = output_scaler.inverse_transform(y_pred_direct.reshape(-1, 1)).flatten()
        y_pred_forecast_denorm = output_scaler.inverse_transform(y_pred_forecast.reshape(-1, 1)).flatten()
        y_pred_combined_denorm = output_scaler.inverse_transform(y_pred_combined.reshape(-1, 1)).flatten()

        # Cap predictions at 0
        y_test_denorm = np.maximum(y_test_denorm, 0)
        y_pred_direct_denorm = np.maximum(y_pred_direct_denorm, 0)
        y_pred_forecast_denorm = np.maximum(y_pred_forecast_denorm, 0)
        y_pred_combined_denorm = np.maximum(y_pred_combined_denorm, 0)

        # Calculate metrics
        print("Calculating metrics...")
        metrics_direct = calculate_metrics(y_test_denorm, y_pred_direct_denorm, f"Direct Forecasting (h={forecast_horizon})")
        metrics_forecast = calculate_metrics(y_test_denorm, y_pred_forecast_denorm, f"Forecast-based (h={forecast_horizon})")
        metrics_combined = calculate_metrics(y_test_denorm, y_pred_combined_denorm, f"Combined Approach (h={forecast_horizon})")

        # Store results
        all_results[forecast_horizon] = {
            'y_test': y_test_denorm,
            'y_pred_direct': y_pred_direct_denorm,
            'y_pred_forecast': y_pred_forecast_denorm,
            'y_pred_combined': y_pred_combined_denorm,
            'metrics_direct': metrics_direct,
            'metrics_forecast': metrics_forecast,
            'metrics_combined': metrics_combined,
            'weights': (direct_weight, forecast_weight)
        }

    except Exception as e:
        print(f"Error processing forecast horizon {forecast_horizon}: {str(e)}")
        continue

# Plot results after processing
plt.figure(figsize=(20, 15))
for idx, forecast_horizon in enumerate(forecast_horizons, 1):
    results = all_results[forecast_horizon]
    y_test = results['y_test']
    y_pred_direct = results['y_pred_direct']
    y_pred_forecast = results['y_pred_forecast']
    y_pred_combined = results['y_pred_combined']
    direct_weight, forecast_weight = results['weights']

    time_steps = range(len(y_test))

    plt.subplot(len(forecast_horizons), 1, idx)
    plt.plot(time_steps, y_test, label='Actual Values', color='black', alpha=0.7)
    plt.plot(time_steps, y_pred_direct, label=f'Direct (w={direct_weight:.2f})', alpha=0.6)
    plt.plot(time_steps, y_pred_forecast, label=f'Forecast (w={forecast_weight:.2f})', alpha=0.6)
    plt.plot(time_steps, y_pred_combined, label='Combined', color='red', alpha=0.8)
    plt.title(f'Forecast Horizon: {forecast_horizon}')
    plt.xlabel('Time Steps')
    plt.ylabel('Streamflow')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print summary of results
print("\nSummary of Results:")
print("="*50)
for horizon in forecast_horizons:
    if horizon in all_results:
        print(f"\nForecast Horizon: {horizon}")
        print("-"*30)
        metrics_direct = all_results[horizon]['metrics_direct']
        metrics_forecast = all_results[horizon]['metrics_forecast']
        metrics_combined = all_results[horizon]['metrics_combined']
        
        print(f"Direct Model - RMSE: {metrics_direct[0]:.2f}, NSE: {metrics_direct[3]:.2f}")
        print(f"Forecast Model - RMSE: {metrics_forecast[0]:.2f}, NSE: {metrics_forecast[3]:.2f}")
        print(f"Combined Model - RMSE: {metrics_combined[0]:.2f}, NSE: {metrics_combined[3]:.2f}") 