from dependencies import (
   # Core data science libraries
   np,
   os,
   joblib,
   pd,
   plt,
   xr,
   tf,
   
   # Sklearn components
   TimeSeriesSplit,
   StandardScaler,
   mean_squared_error,
   mean_absolute_error,
   r2_score,
   
   # Keras/TensorFlow components
   Sequential,
   Model,
   LSTM,
   Dense,
   Dropout,
   InputLayer,
   Adam,
   MeanSquaredError,
   RootMeanSquaredError,
   
   # Local utility functions
   preprocessing_data,
   prepare_direct_data,
   prepare_forecast_data,
   calculate_weights,
   calculate_metrics
)

new_working_directory = r"C:\Users\NVN\Master_Thesis\Preprocessed_data\Final_Model_Codes"
os.chdir(new_working_directory)

np.random.seed(123)
tf.random.set_seed(123)

# Define file paths and model save directory
input_file_path = r"C:\Users\NVN\Master_Thesis\Preprocessed_data\Iori\Preprocessed_data_GE.nc"
model_save_dir = r"C:\Users\NVN\Master_Thesis\Models\Iori\LSTM"

# Prepare data
X_train, y_train, X_val, y_val, X_test, y_test, ds, output_scaler = preprocessing_data(input_file_path, model_save_dir)

# Build model function (modified to specify input shape)
def build_lstm_model(input_shape, units1=75, units2=130, dropout_rate=0.25315, learning_rate=0.000456):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        LSTM(units=units1, activation='tanh', return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units2, activation='tanh'),
        Dropout(dropout_rate),
        Dense(units=1, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])
    return model

# Define forecast horizons to test
forecast_horizons = [1, 2, 4, 12, 24]
all_results = {}

for forecast_horizon in forecast_horizons:
    print(f"\nProcessing forecast horizon: {forecast_horizon}")
    
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

        # Reshape data for LSTM
        print("Reshaping data...")
        X_direct_train = X_direct_train.reshape((X_direct_train.shape[0], X_direct_train.shape[1], X_direct_train.shape[2]))
        X_direct_val = X_direct_val.reshape((X_direct_val.shape[0], X_direct_val.shape[1], X_direct_val.shape[2]))
        X_direct_test = X_direct_test.reshape((X_direct_test.shape[0], X_direct_test.shape[1], X_direct_test.shape[2]))

        X_forecast_train = X_forecast_train.reshape((X_forecast_train.shape[0], X_forecast_train.shape[1], X_forecast_train.shape[2]))
        X_forecast_val = X_forecast_val.reshape((X_forecast_val.shape[0], X_forecast_val.shape[1], X_forecast_val.shape[2]))
        X_forecast_test = X_forecast_test.reshape((X_forecast_test.shape[0], X_forecast_test.shape[1], X_forecast_test.shape[2]))

        # Create and train models
        print("Creating and training models...")
        input_shape_direct = (X_direct_train.shape[1], X_direct_train.shape[2])
        input_shape_forecast = (X_forecast_train.shape[1], X_forecast_train.shape[2])

        # Initialize models
        direct_model = build_lstm_model(input_shape=input_shape_direct)
        forecast_model = build_lstm_model(input_shape=input_shape_forecast)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train models
        print("Training direct model...")
        direct_model.fit(
            X_direct_train, y_direct_train,
            validation_data=(X_direct_val, y_direct_val),
            epochs=71,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )

        print("Training forecast model...")
        forecast_model.fit(
            X_forecast_train, y_forecast_train,
            validation_data=(X_forecast_val, y_forecast_val),
            epochs=71,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )

        # Save models
        direct_model_path = os.path.join(model_save_dir, f'direct_lstm_model_FH{forecast_horizon}.h5')
        forecast_model_path = os.path.join(model_save_dir, f'forecast_lstm_model_FH{forecast_horizon}.h5')
        direct_model.save(direct_model_path)
        forecast_model.save(forecast_model_path)

        # Generate predictions
        print("Generating predictions...")
        y_pred_direct = direct_model.predict(X_direct_test)
        y_pred_forecast = forecast_model.predict(X_forecast_test)

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
        
        print(f"Direct Model - RMSE: {metrics_direct[1]:.2f}, NSE: {metrics_direct[4]:.2f}")
        print(f"Forecast Model - RMSE: {metrics_forecast[1]:.2f}, NSE: {metrics_forecast[4]:.2f}")
        print(f"Combined Model - RMSE: {metrics_combined[1]:.2f}, NSE: {metrics_combined[4]:.2f}")