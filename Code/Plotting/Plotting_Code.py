__all__ = ['ELM', 'plot_combined_model']

from project_paths import get_models_dir
from Code.dependencies import (
   # Core data science libraries
   os,
   np,
   plt,
   joblib,
   tf,
   xr,
   
   # Keras/TensorFlow components
   load_model,
   
   # Statistical metrics
   sklearn_mse,  # mean_squared_error with alias
   mae,          # mean_absolute_error with alias
   pearsonr,
   skew,
   kurtosis,
   
   # Matplotlib date utilities
   YearLocator,
   DateFormatter,
   AutoDateLocator,
   
   # Local utility functions
   nash_sutcliffe_efficiency,
   calculate_metrics,
   prepare_forecast_data,
   prepare_direct_data,
   
   # Visualization functions
   load_data,
   load_model_file,
   get_combined_predictions,
   plot_combined_model,
   plot_combined_models_comparison,
   plot_scatter_comparisons,
   plot_residual_analysis
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

np.random.seed(123)
tf.random.set_seed(123)

model_dirs = {
    'Iori_CNN': os.path.join(get_models_dir(), 'Iori', 'CNN'),
    'Iori_LSTM': os.path.join(get_models_dir(), 'Iori', 'LSTM'),
    'Iori_ELM': os.path.join(get_models_dir(), 'Iori', 'ELM'),
    'Secchia_CNN': os.path.join(get_models_dir(), 'Secchia', 'CNN'),
    'Secchia_LSTM': os.path.join(get_models_dir(), 'Secchia', 'LSTM'),
    'Secchia_ELM': os.path.join(get_models_dir(), 'Secchia', 'ELM')
}

weights_dir = os.path.join(get_models_dir(), 'Weights')
forecast_horizons = [1, 2, 4, 12, 24]

#################  Plot for all Hydrographs (single) #######################
for basin in ['Iori', 'Secchia']:
    for model_type in ['CNN', 'LSTM', 'ELM']:
        model_key = f'{basin}_{model_type}'
        if model_key in model_dirs:
            print(f"\nProcessing {model_key}")
            model_dir = model_dirs[model_key]
            
            # Create and show plot
            fig = plot_combined_model(
                model_dir=model_dir,
                weights_dir=weights_dir,
                forecast_horizons=forecast_horizons,
                model_type=model_type.lower()
            )
            
            fig.suptitle(f'{basin} Basin - {model_type} Combined Model', fontsize=16)
            plt.show()

# Calculate metrics for all models#############

all_metrics = {}

# Loop through basins and models
for basin in ['Iori', 'Secchia']:
    all_metrics[basin] = {}
    print(f"\n{'='*50}")
    print(f"Metrics for {basin} Basin")
    print(f"{'='*50}")
    
    for model_type in ['CNN', 'LSTM', 'ELM']:
        all_metrics[basin][model_type] = {}
        model_key = f'{basin}_{model_type}'
        
        if model_key in model_dirs:
            print(f"\n{'-'*20} {model_type} Model {'-'*20}")
            model_dir = model_dirs[model_key]
            
            # Load data
            X_test, y_test, output_scaler, test_dates, ds = load_data(model_dir)
            
            # Calculate metrics for each forecast horizon
            for horizon in forecast_horizons:
                print(f"\nForecast Horizon: {horizon}")
                
                # In your main loop, modify this part:
                try:
                    # Get predictions
                    y_true_denorm, y_pred_denorm = get_combined_predictions(
                        model_dir, weights_dir, 
                        X_test, y_test,
                        output_scaler, horizon, model_type.lower(), ds
                    )
    
                    # Calculate metrics - now properly unpack the tuple
                    rmse, mae_value, pearson_r, nse, mse = calculate_metrics(y_true_denorm, y_pred_denorm)
    
                    # Store metrics in dictionary format
                    metrics = {
                        'RMSE': rmse,
                        'MAE': mae_value,
                        "Pearson's r": pearson_r,
                        'NSE': nse,
                        'MSE': mse
                    }
    
                    all_metrics[basin][model_type][horizon] = metrics
    
                    # Print metrics in a formatted way
                    print(f"RMSE: {rmse:.4f}")
                    print(f"MAE: {mae_value:.4f}")
                    print(f"Pearson r: {pearson_r:.4f}")
                    print(f"NSE: {nse:.4f}")
                    print(f"MSE: {mse:.4f}")
                    
                except Exception as e:
                    print(f"Error processing horizon {horizon}: {str(e)}")
                    all_metrics[basin][model_type][horizon] = None

# Print summary table for each metric
metrics_to_display = ['RMSE', 'MAE', "Pearson's r", 'NSE', 'MSE']

for metric_name in metrics_to_display:
    print(f"\n{'='*100}")
    print(f"Summary of {metric_name}")
    print(f"{'='*100}")
    
    # Print header
    print(f"{'Basin':<10} {'Model':<6} " + " ".join(f"FH{h:<6}" for h in forecast_horizons))
    print("-" * 100)
    
    # Print values
    for basin in ['Iori', 'Secchia']:
        for model_type in ['CNN', 'LSTM', 'ELM']:
            row = f"{basin:<10} {model_type:<6} "
            for horizon in forecast_horizons:
                if (all_metrics[basin][model_type][horizon] is not None and 
                    metric_name in all_metrics[basin][model_type][horizon]):
                    value = all_metrics[basin][model_type][horizon][metric_name]
                    row += f"{value:7.4f} "
                else:
                    row += "   N/A  "
            print(row)


################ Plot combined results ################


for basin in ['Iori', 'Secchia']:
    fig = plot_combined_models_comparison(
        model_dirs=model_dirs,  
        weights_dir=weights_dir,  
        basin=basin,
        selected_forecast_horizons=[1, 4, 24]
    )
    plt.show()


################# Scatter Plots #########################


selected_forecast_horizons = [1, 4, 24]
for basin in ['Iori', 'Secchia']:
    fig = plot_scatter_comparisons(
        model_dirs=model_dirs,
        weights_dir=weights_dir,
        basin=basin, 
        selected_forecast_horizons=selected_forecast_horizons
    )
    plt.show()

######### Residuals Plot #########

selected_forecast_horizons = [1, 4, 24]
for basin in ['Iori', 'Secchia']:
    fig = plot_residual_analysis(
        model_dirs=model_dirs,
        weights_dir=weights_dir,
        basin=basin,
        selected_forecast_horizons=selected_forecast_horizons
    )
    plt.show()