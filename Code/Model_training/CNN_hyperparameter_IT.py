import os
from project_paths import get_code_dir, get_data_dir, get_models_dir

from Code.dependencies import (
    # Core data science libraries
    np,
    pd,
    plt,
    xr,
    
    # Sklearn components
    TimeSeriesSplit,
    StandardScaler,
    mean_squared_error,
    mean_absolute_error,
    
    # TensorFlow and Keras components
    tf,
    Sequential,
    Model,
    
    # Keras layers
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    ReLU,
    Input,
    Concatenate,
    GlobalAveragePooling1D,
    
    # Keras callbacks
    ModelCheckpoint,
    EarlyStopping,
    
    # Hyperparameter tuning
    kt
)

input_file_path = os.path.join(get_data_dir(), 'Secchia', 'Preprocessed_data_IT.nc')
ds = xr.open_dataset(input_file_path)

# Select input features and output feature
input_features = ds[['weighted_tp', 'weighted_t2m']].to_dataframe().dropna()
output_feature = ds['discharge'].to_dataframe().dropna() 

# Align the input and output features
input_features = input_features.loc[output_feature.index]
output_feature = output_feature.loc[input_features.index]

# Merge discharge with input_features to add lagged features
input_features['discharge'] = output_feature

# Add lagged outflow features to the input features
def add_lagged_features(df, target_column, lags):
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    return df

lags = [1, 2, 3]
input_features = add_lagged_features(input_features, 'discharge', lags).dropna()

# Separate the features and target again after adding lagged features
output_feature = input_features['discharge']
input_features = input_features.drop(columns=['discharge'])

# Align the input and output features again after adding lagged features
input_features = input_features.loc[output_feature.index]
output_feature = output_feature.loc[input_features.index]

forecast_horizon = 4

# Split data
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)
splits = list(tscv.split(input_features))
train_val_idx, test_idx = splits[-1]
train_idx, val_idx = splits[-2]

# Normalize the data separately for training, validation, and test sets
scaler = StandardScaler()
X_train = scaler.fit_transform(input_features.iloc[train_idx])
X_val = scaler.transform(input_features.iloc[val_idx])
X_test = scaler.transform(input_features.iloc[test_idx])

output_scaler = StandardScaler()
y_train = output_scaler.fit_transform(output_feature.iloc[train_idx].values.reshape(-1, 1)).flatten()
y_val = output_scaler.transform(output_feature.iloc[val_idx].values.reshape(-1, 1)).flatten()
y_test = output_scaler.transform(output_feature.iloc[test_idx].values.reshape(-1, 1)).flatten()

# Prepare data for direct and forecast-based approaches
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

# Prepare both types of data
X_direct_train, y_direct_train = prepare_direct_data(X_train, y_train, forecast_horizon)
X_direct_val, y_direct_val = prepare_direct_data(X_val, y_val, forecast_horizon)
X_direct_test, y_direct_test = prepare_direct_data(X_test, y_test, forecast_horizon)

X_forecast_train, y_forecast_train = prepare_forecast_data(X_train, y_train, forecast_horizon, ds)
X_forecast_val, y_forecast_val = prepare_forecast_data(X_val, y_val, forecast_horizon, ds)
X_forecast_test, y_forecast_test = prepare_forecast_data(X_test, y_test, forecast_horizon, ds)

# Reshape input data for Conv1D (samples, time steps, features)
X_direct_train = X_direct_train.reshape((X_direct_train.shape[0], X_direct_train.shape[1], X_direct_train.shape[2]))
X_direct_val = X_direct_val.reshape((X_direct_val.shape[0], X_direct_val.shape[1], X_direct_val.shape[2]))
X_direct_test = X_direct_test.reshape((X_direct_test.shape[0], X_direct_test.shape[1], X_direct_test.shape[2]))

X_forecast_train = X_forecast_train.reshape((X_forecast_train.shape[0], X_forecast_train.shape[1], X_forecast_train.shape[2]))
X_forecast_val = X_forecast_val.reshape((X_forecast_val.shape[0], X_forecast_val.shape[1], X_forecast_val.shape[2]))
X_forecast_test = X_forecast_test.reshape((X_forecast_test.shape[0], X_forecast_test.shape[1], X_forecast_test.shape[2]))

# Define the CNN model with parallel convolution branches using different kernel sizes
def build_model(hp):
    filters1 = hp.Int('filters1', min_value=32, max_value=128, step=32)
    filters2 = hp.Int('filters2', min_value=16, max_value=64, step=16)
    filters3 = hp.Int('filters3', min_value=64, max_value=256, step=64)
    kernel_size1 = hp.Choice('kernel_size1', values=[3, 5, 7])
    kernel_size2 = hp.Choice('kernel_size2', values=[3, 5, 7])
    kernel_size3 = hp.Choice('kernel_size3', values=[3, 5, 7])
    dense_units1 = hp.Int('dense_units1', min_value=128, max_value=512, step=128)
    dense_units2 = hp.Int('dense_units2', min_value=64, max_value=256, step=64)
    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    
    inputs = Input(shape=(X_direct_train.shape[1], X_direct_train.shape[2]))
    
    conv_outputs = []
    for kernel_size in [kernel_size1, kernel_size2, kernel_size3]:
        conv = Conv1D(filters=filters1, kernel_size=kernel_size, padding='same')(inputs)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        
        conv = Conv1D(filters=filters2, kernel_size=kernel_size, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        conv = Dropout(dropout_rate)(conv)
        
        conv = Conv1D(filters=filters3, kernel_size=kernel_size, padding='same')(conv)
        conv = BatchNormalization()(conv)
        conv = ReLU()(conv)
        
        conv_outputs.append(conv)
    
    if len(conv_outputs) > 1:
        x = Concatenate()(conv_outputs)
    else:
        x = conv_outputs[0]
    
    x = GlobalAveragePooling1D()(x)
    
    x = Dense(dense_units1, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mae'])
    
    return model

# Set up the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,
    executions_per_trial=1,
    directory='my_dir',
    project_name='cnn_hyperparameter_tuning'
)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Run the tuner search
tuner.search(X_direct_train, y_direct_train,
             epochs=50,
             validation_data=(X_direct_val, y_direct_val),
             callbacks=[early_stopping])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
print("The hyperparameter search is complete. The optimal hyperparameters are:")
print(f"filters1: {best_hps.get('filters1')}")
print(f"filters2: {best_hps.get('filters2')}")
print(f"filters3: {best_hps.get('filters3')}")
print(f"kernel_size1: {best_hps.get('kernel_size1')}")
print(f"kernel_size2: {best_hps.get('kernel_size2')}")
print(f"kernel_size3: {best_hps.get('kernel_size3')}")
print(f"dense_units1: {best_hps.get('dense_units1')}")
print(f"dense_units2: {best_hps.get('dense_units2')}")
print(f"dropout_rate: {best_hps.get('dropout_rate')}")

# Build the model with the best hyperparameters
best_model = build_model(best_hps)

# Train the best model
history = best_model.fit(X_direct_train, y_direct_train,
                         validation_data=(X_direct_val, y_direct_val),
                         epochs=100,
                         batch_size=32,
                         callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_mae = best_model.evaluate(X_direct_test, y_direct_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Make predictions with the best model
y_pred = best_model.predict(X_direct_test)

# Denormalize predictions
y_test_denorm = output_scaler.inverse_transform(y_direct_test.reshape(-1, 1)).flatten()
y_pred_denorm = output_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Calculate metrics
mse = mean_squared_error(y_test_denorm, y_pred_denorm)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
correlation = np.corrcoef(y_test_denorm, y_pred_denorm)[0, 1]
nse = 1 - (np.sum((y_test_denorm - y_pred_denorm) ** 2) / np.sum((y_test_denorm - np.mean(y_test_denorm)) ** 2))

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'Correlation: {correlation:.2f}')
print(f'NSE: {nse:.2f}')

# Plot the actual values vs the predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_denorm, label='Actual Values')
plt.plot(y_pred_denorm, label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Streamflow')
plt.legend()
plt.show()