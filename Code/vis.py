# Core data science libraries
import os
import numpy as np
import xarray as xr
import joblib
import matplotlib.pyplot as plt

# Keras/TensorFlow components
from tensorflow.keras.models import load_model

# Statistical functions
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import mean_absolute_error as mae

# Matplotlib date utilities
from matplotlib.dates import YearLocator, DateFormatter, AutoDateLocator

# Local utility functions 
from .utils import (
    prepare_direct_data,
    prepare_forecast_data,
    calculate_metrics
)

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
        self.output_weights = np.dot(np.linalg.pinv(H), y)

    def predict(self, X):
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        return np.dot(H, self.output_weights)

def load_data(model_dir):
    """
    Load test data, scaler, dates, and dataset from the specified directory
    """
    X_test = np.load(os.path.join(model_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(model_dir, 'y_test.npy'))
    output_scaler = joblib.load(os.path.join(model_dir, 'output_scaler.pkl'))
    test_dates = np.load(os.path.join(model_dir, 'test_dates.npy'), allow_pickle=True)
    
    # Load the original dataset
    if 'Iori' in model_dir:
        ds = xr.open_dataset(r"C:\Users\NVN\Master_Thesis\Preprocessed_data\Iori\Preprocessed_data_GE.nc")
    else:  # Secchia
        ds = xr.open_dataset(r"C:\Users\NVN\Master_Thesis\Preprocessed_data\Secchia\Preprocessed_data_IT.nc")
    
    return X_test, y_test, output_scaler, test_dates, ds


def load_model_file(model_path):
    """
    Load model from file, handling different file formats
    """
    if model_path.endswith('.h5'):
        return load_model(model_path)
    elif model_path.endswith('.joblib'):
        return joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")


def get_combined_predictions(model_dir, weights_dir, X_test, y_test, output_scaler, forecast_horizon, model_type, ds):
    """
    Get predictions using exactly the same approach as in training, including forecast data
    """
    # Load models
    direct_model_path = os.path.join(model_dir, 
                                    f'direct_{model_type.lower()}_model_FH{forecast_horizon}.{"h5" if model_type != "elm" else "joblib"}')
    forecast_model_path = os.path.join(model_dir, 
                                      f'forecast_{model_type.lower()}_model_FH{forecast_horizon}.{"h5" if model_type != "elm" else "joblib"}')
    
    direct_model = load_model_file(direct_model_path)
    forecast_model = load_model_file(forecast_model_path)
    
    # Load weights
    weights_path = os.path.join(weights_dir, f'combined_weights_FH{forecast_horizon}.joblib')
    weights = joblib.load(weights_path)
    direct_weight, forecast_weight = weights['direct_weight'], weights['forecast_weight']

    # Prepare data exactly as in training, including the dataset for forecast data
    X_direct, y_direct = prepare_direct_data(X_test, y_test, forecast_horizon)
    X_forecast, y_forecast = prepare_forecast_data(X_test, y_test, forecast_horizon, ds)  # Now passing ds

    # For ELM models, reshape the input
    if model_type.lower() == 'elm':
        X_direct = X_direct.reshape((X_direct.shape[0], -1))
        X_forecast = X_forecast.reshape((X_forecast.shape[0], -1))

    # Get predictions
    y_pred_direct = direct_model.predict(X_direct)
    y_pred_forecast = forecast_model.predict(X_forecast)

    # Combine predictions using weights
    y_pred_combined = (direct_weight * y_pred_direct + forecast_weight * y_pred_forecast)

    # Denormalize predictions
    y_true_denorm = output_scaler.inverse_transform(y_direct.reshape(-1, 1)).flatten()
    y_pred_denorm = output_scaler.inverse_transform(y_pred_combined.reshape(-1, 1)).flatten()

    return y_true_denorm, y_pred_denorm

def plot_combined_model(model_dir, weights_dir, forecast_horizons=[1, 2, 4, 12, 24], model_type='cnn'):
    """
    Plot predictions with correct sequence alignment and forecast data, with updated appearance
    """
    # Load all necessary data, including the dataset
    X_test, y_test, output_scaler, test_dates, ds = load_data(model_dir)
    
    # Set color based on model type
    color = 'blue' if model_type.lower() == 'lstm' else 'green' if model_type.lower() == 'cnn' else 'red'
    
    n_horizons = len(forecast_horizons)
    fig, axes = plt.subplots(n_horizons, 1, figsize=(15, 5*n_horizons))
    if n_horizons == 1:
        axes = [axes]

    for idx, horizon in enumerate(forecast_horizons):
        try:
            # Get predictions using the corrected function with dataset
            y_true_denorm, y_pred_denorm = get_combined_predictions(
                model_dir, weights_dir, 
                X_test, y_test,
                output_scaler, horizon, model_type, ds
            )

            # Calculate metrics
            metrics = calculate_metrics(y_true_denorm, y_pred_denorm)
            if metrics is None:
                print(f"Skipping horizon {horizon} due to error in metric calculation")
                continue

            # Align dates correctly with predictions
            time_steps = test_dates[horizon:horizon + len(y_true_denorm)]

            # Plot both series
            axes[idx].plot(time_steps, y_true_denorm, 
                          color='black', label='Actual', alpha=0.6)
            axes[idx].plot(time_steps, y_pred_denorm, 
                         color=color, label=f'{model_type.upper()} FH{horizon}', alpha=0.6)
            
            # Add title inside the subplot
            axes[idx].text(0.5, 0.9, f'Forecast Horizon: {horizon}', 
                           horizontalalignment='center', 
                           verticalalignment='center', 
                           transform=axes[idx].transAxes, 
                           fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            
            # Format the x-axis with dates
            axes[idx].xaxis.set_major_locator(YearLocator())
            axes[idx].xaxis.set_major_formatter(DateFormatter('%Y'))
            if idx == len(forecast_horizons) - 1:
                axes[idx].xaxis.set_minor_locator(AutoDateLocator())
                axes[idx].xaxis.set_minor_formatter(DateFormatter(''))
                axes[idx].set_xlabel('Year')
            
            # Rotate and align the tick labels so they look better
            for label in axes[idx].get_xticklabels(which='both'):
                label.set_rotation(0)
                label.set_horizontalalignment('center')
            
            # Customize subplot
            axes[idx].set_ylabel('Streamflow')
            axes[idx].legend(loc='upper right')  # Force legend to be in the top right corner
            axes[idx].grid(True)

        except Exception as e:
            print(f"Error processing horizon {horizon}: {str(e)}")
            continue

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to add more space below the last subplot
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05)  # Add more space between subplots
    return fig

def plot_combined_models_comparison(model_dirs, weights_dir, basin='Secchia', selected_forecast_horizons=[1, 4, 24]):
    
    # Create figure
    fig, axes = plt.subplots(len(selected_forecast_horizons), 1, figsize=(15, 5*len(selected_forecast_horizons)))
    if len(selected_forecast_horizons) == 1:
        axes = [axes]

    # Load data for each model type
    model_data = {}
    for model_type in ['CNN', 'LSTM', 'ELM']:
        model_key = f'{basin}_{model_type}'
        if model_key in model_dirs:
            model_dir = model_dirs[model_key]
            X_test, y_test, output_scaler, test_dates, ds = load_data(model_dir)
            model_data[model_type] = {
                'dir': model_dir,
                'X_test': X_test,
                'y_test': y_test,
                'output_scaler': output_scaler,
                'test_dates': test_dates,
                'ds': ds
            }

    # Colors for each model
    colors = {'CNN': 'green', 'LSTM': 'blue', 'ELM': 'red'}

    # Plot for each forecast horizon
    for idx, horizon in enumerate(selected_forecast_horizons):
        # First plot actual values using CNN data
        y_true_denorm, y_pred_denorm = get_combined_predictions(
            model_data['CNN']['dir'], weights_dir,
            model_data['CNN']['X_test'], model_data['CNN']['y_test'],
            model_data['CNN']['output_scaler'], horizon, 'cnn', model_data['CNN']['ds']
        )
        time_steps = model_data['CNN']['test_dates'][horizon:horizon + len(y_true_denorm)]
        
        # Plot actual values
        axes[idx].plot(time_steps, y_true_denorm, 
                      label='Actual', color='black', linestyle='-', linewidth=2, alpha=0.8)

        # Plot predictions for each model
        for model_type in ['CNN', 'LSTM', 'ELM']:
            try:
                y_true_denorm, y_pred_denorm = get_combined_predictions(
                    model_data[model_type]['dir'], weights_dir,
                    model_data[model_type]['X_test'], model_data[model_type]['y_test'],
                    model_data[model_type]['output_scaler'], horizon, model_type.lower(), 
                    model_data[model_type]['ds']
                )
                
                # Ensure predictions are non-negative
                y_pred_denorm = np.maximum(y_pred_denorm, 0)
                
                # Plot predictions
                time_steps = model_data[model_type]['test_dates'][horizon:horizon + len(y_pred_denorm)]
                axes[idx].plot(time_steps, y_pred_denorm, 
                             label=model_type, color=colors[model_type], linewidth=2, alpha=0.8)
                
            except Exception as e:
                print(f"Error plotting {model_type} for horizon {horizon}: {str(e)}")

        # Customize subplot
        axes[idx].text(0.5, 0.95, f'Forecast Horizon: {horizon} weeks', 
                      fontsize=12, ha='center', va='top', 
                      transform=axes[idx].transAxes, 
                      bbox=dict(facecolor='white', alpha=0.8))
        axes[idx].set_ylabel('Streamflow')
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True)

        # Format x-axis
        axes[idx].xaxis.set_major_locator(YearLocator())
        axes[idx].xaxis.set_major_formatter(DateFormatter('%Y'))
        if idx == len(selected_forecast_horizons) - 1:
            axes[idx].xaxis.set_minor_locator(AutoDateLocator())
            axes[idx].xaxis.set_minor_formatter(DateFormatter(''))
            axes[idx].set_xlabel('Year')
        
        # Rotate and align tick labels
        for label in axes[idx].get_xticklabels(which='both'):
            label.set_rotation(0)
            label.set_horizontalalignment('center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05)
    fig.suptitle(f'{basin} Basin - Model Comparisons for Selected Forecast Horizons', fontsize=16)
    return fig

def plot_scatter_comparisons(model_dirs, weights_dir, basin='Secchia', selected_forecast_horizons=[1, 4, 24]):
    # Create figure
    fig, axes = plt.subplots(len(selected_forecast_horizons), 3, figsize=(15, 5*len(selected_forecast_horizons)))

    # Define model configurations
    model_configs = [
        ('CNN', 'green'),
        ('LSTM', 'blue'),
        ('ELM', 'red')
    ]

    # Load data for each model type
    model_data = {}
    for model_type, _ in model_configs:
        model_key = f'{basin}_{model_type}'
        if model_key in model_dirs:
            model_dir = model_dirs[model_key]
            X_test, y_test, output_scaler, test_dates, ds = load_data(model_dir)
            model_data[model_type] = {
                'dir': model_dir,
                'X_test': X_test,
                'y_test': y_test,
                'output_scaler': output_scaler,
                'ds': ds
            }

    # Plot for each forecast horizon and model
    for row_idx, horizon in enumerate(selected_forecast_horizons):
        for col_idx, (model_type, color) in enumerate(model_configs):
            try:
                # Get predictions
                y_true_denorm, y_pred_denorm = get_combined_predictions(
                    model_data[model_type]['dir'], weights_dir,
                    model_data[model_type]['X_test'], 
                    model_data[model_type]['y_test'],
                    model_data[model_type]['output_scaler'], 
                    horizon, model_type.lower(), 
                    model_data[model_type]['ds']
                )

                # Cap predictions at 0 for LSTM and ELM
                if model_type in ['LSTM', 'ELM']:
                    y_pred_denorm = np.maximum(y_pred_denorm, 0)

                # Calculate Pearson's r
                pearson_r, _ = pearsonr(y_true_denorm, y_pred_denorm)

                # Create scatter plot
                axes[row_idx, col_idx].scatter(y_true_denorm, y_pred_denorm, color=color, alpha=0.6)
                axes[row_idx, col_idx].plot(
                    [y_true_denorm.min(), y_true_denorm.max()], 
                    [y_true_denorm.min(), y_true_denorm.max()], 
                    'k--', lw=2
                )
                axes[row_idx, col_idx].set_title(f'{model_type} FH{horizon}')
                axes[row_idx, col_idx].set_xlabel('Actual Precip. mm/w')
                axes[row_idx, col_idx].set_ylabel('Predicted Precip. mm/w')
                axes[row_idx, col_idx].text(
                    0.05, 0.95, 
                    f'r = {pearson_r:.2f}', 
                    transform=axes[row_idx, col_idx].transAxes, 
                    fontsize=12, 
                    verticalalignment='top', 
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                axes[row_idx, col_idx].grid(True)

            except Exception as e:
                print(f"Error plotting {model_type} for horizon {horizon}: {str(e)}")
                axes[row_idx, col_idx].text(
                    0.5, 0.5, 
                    'Error plotting data', 
                    ha='center', va='center'
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9, bottom=0.05)
    fig.suptitle(f'{basin} Basin - Predicted vs Actual Scatter Plots for Selected Forecast Horizons', fontsize=16)
    return fig

def plot_residual_analysis(model_dirs, weights_dir, basin='Iori', selected_forecast_horizons=[1, 2, 4, 12, 24]):
    
    # Create figure
    fig, axes = plt.subplots(len(selected_forecast_horizons), 3, figsize=(15, 5*len(selected_forecast_horizons)))

    # Define model configurations
    model_configs = [
        ('CNN', 'green'),
        ('LSTM', 'blue'),
        ('ELM', 'red')
    ]

    # Load data for each model type
    model_data = {}
    for model_type, _ in model_configs:
        model_key = f'{basin}_{model_type}'
        if model_key in model_dirs:
            model_dir = model_dirs[model_key]
            X_test, y_test, output_scaler, test_dates, ds = load_data(model_dir)
            model_data[model_type] = {
                'dir': model_dir,
                'X_test': X_test,
                'y_test': y_test,
                'output_scaler': output_scaler,
                'ds': ds
            }

    # Plot for each forecast horizon and model
    for row_idx, horizon in enumerate(selected_forecast_horizons):
        for col_idx, (model_type, color) in enumerate(model_configs):
            try:
                # Get predictions
                y_true_denorm, y_pred_denorm = get_combined_predictions(
                    model_data[model_type]['dir'], weights_dir,
                    model_data[model_type]['X_test'], 
                    model_data[model_type]['y_test'],
                    model_data[model_type]['output_scaler'], 
                    horizon, model_type.lower(), 
                    model_data[model_type]['ds']
                )

                # Cap predictions at 0 for LSTM and ELM
                if model_type in ['LSTM', 'ELM']:
                    y_pred_denorm = np.maximum(y_pred_denorm, 0)

                # Calculate residuals
                residuals = y_true_denorm - y_pred_denorm

                # Calculate residual metrics
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                mae_residual = mae(y_true_denorm, y_pred_denorm)
                rmse_residual = np.sqrt(sklearn_mse(y_true_denorm, y_pred_denorm))
                skew_residual = skew(residuals)
                kurt_residual = kurtosis(residuals)

                # Print residual metrics
                print(f"\n{model_type} FH{horizon} Residual Metrics:")
                print(f"  Mean Residual: {mean_residual:.4f}")
                print(f"  Std Residual: {std_residual:.4f}")
                print(f"  MAE: {mae_residual:.4f}")
                print(f"  RMSE: {rmse_residual:.4f}")
                print(f"  Skewness: {skew_residual:.4f}")
                print(f"  Kurtosis: {kurt_residual:.4f}")

                # Create residual plot
                axes[row_idx, col_idx].scatter(y_true_denorm, residuals, color=color, alpha=0.6)
                axes[row_idx, col_idx].axhline(0, color='red', linestyle='--')
                axes[row_idx, col_idx].set_title(f'Residuals {model_type} FH{horizon}')
                axes[row_idx, col_idx].set_xlabel('Actual Precip. mm/w')
                axes[row_idx, col_idx].set_ylabel('Residual Precip. mm/w')
                
                # Add text box with key metrics
                metrics_text = f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}'
                axes[row_idx, col_idx].text(
                    0.05, 0.95, 
                    metrics_text,
                    transform=axes[row_idx, col_idx].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8)
                )
                
                axes[row_idx, col_idx].grid(True)

            except Exception as e:
                print(f"Error plotting {model_type} for horizon {horizon}: {str(e)}")
                axes[row_idx, col_idx].text(
                    0.5, 0.5, 
                    'Error plotting data', 
                    ha='center', va='center'
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9, bottom=0.05)
    fig.suptitle(f'{basin} Basin - Residual Plots for Selected Forecast Horizons', fontsize=16)
    return fig