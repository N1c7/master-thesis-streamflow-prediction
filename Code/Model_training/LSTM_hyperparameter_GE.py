import os
from project_paths import get_code_dir, get_data_dir, get_models_dir

from dependencies import (
   # Core data science libraries
   tf,
   pd, 
   np,
   plt,
   xr,
   
   # Keras/TensorFlow components
   Sequential,
   InputLayer,
   LSTM, 
   Dense,
   Dropout,
   ModelCheckpoint,
   MeanSquaredError,
   RootMeanSquaredError,
   Adam,
   
   # Optimization and hyperparameter tuning
   gp_minimize,
   Real,
   Integer, 
   Categorical,
   
   # Sklearn components
   TimeSeriesSplit,
   mean_squared_error
)

input_file_path = os.path.join(get_data_dir(), 'Iori', 'Preprocessed_data_GE.nc')
ds = xr.open_dataset(input_file_path)

# Select input features and output feature
input_features = ds[['weighted_t2m', 'weighted_tp']].to_dataframe().dropna()
output_feature = ds['discharge'].to_dataframe().dropna()

# Align the input and output features
input_features = input_features.loc[output_feature.index]
output_feature = output_feature.loc[input_features.index]

# Normalize the data
input_features_normalized = (input_features - input_features.min()) / (input_features.max() - input_features.min())
output_feature_normalized = (output_feature - output_feature.min()) / (output_feature.max() - output_feature.min())

# Define the window size
window_size = 52

# Function to create sequences of data
def df_to_X_y(input_features, output_feature, window_size):
    input_features_as_np = input_features.to_numpy()
    output_feature_as_np = output_feature.to_numpy()
    X, y = [], []
    for i in range(len(input_features_as_np) - window_size):
        X.append(input_features_as_np[i:i + window_size])
        y.append(output_feature_as_np[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Create sequences of input and output data
X, y = df_to_X_y(input_features_normalized, output_feature_normalized, window_size)

# Define the model building function with two LSTM layers and gradient clipping
def build_model(units1, units2, activation, learning_rate, dropout_rate):
    model = Sequential([
        InputLayer(input_shape=(window_size, X.shape[2])),
        LSTM(units=units1, activation=activation, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units2, activation=activation),
        Dropout(dropout_rate),
        Dense(units=1, activation='linear')
    ])
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=[RootMeanSquaredError()])
    return model

# Define the search space
search_space = [
    Integer(low=10, high=150, name='units1'),
    Integer(low=10, high=150, name='units2'),
    Categorical(categories=['relu', 'tanh'], name='activation'),
    Real(low=1e-5, high=1e-3, name='learning_rate'), 
    Integer(low=8, high=32, name='batch_size'),
    Integer(low=10, high=75, name='epochs'),
    Real(low=0.1, high=0.5, name='dropout_rate')
]

# Define the objective function
def objective_fn(params):
    units1, units2, activation, learning_rate, batch_size, epochs, dropout_rate = params
    print(f"Testing parameters: units1={units1}, units2={units2}, activation={activation}, learning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}, dropout_rate={dropout_rate}")
    model = build_model(units1, units2, activation, learning_rate, dropout_rate)
    tscv = TimeSeriesSplit(n_splits=3)
    val_losses = []
    for train_index, val_index in tscv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        cp = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[cp], verbose=1)
        val_loss = min(history.history['val_loss'])
        if np.isnan(val_loss):
            print("NaN value detected in validation loss. Returning infinity.")
            return np.inf
        val_losses.append(val_loss)
    avg_val_loss = np.mean(val_losses)
    print(f"Average validation loss: {avg_val_loss}")
    return avg_val_loss

# Run the optimization
res = gp_minimize(
    func=objective_fn, 
    dimensions=search_space, 
    n_calls=20, 
    random_state=42,
    acq_func="EI",
    n_initial_points=5,
    noise=1e-10
)

# Print the best hyperparameters
print("Best hyperparameters:")
print(f"Number of units in first LSTM layer: {res.x[0]}")
print(f"Number of units in second LSTM layer: {res.x[1]}")
print(f"Activation function: {res.x[2]}")
print(f"Learning rate: {res.x[3]}")
print(f"Batch size: {res.x[4]}")
print(f"Epochs: {res.x[5]}")
print(f"Dropout rate: {res.x[6]}")

# Load the best model
model = tf.keras.models.load_model('best_model.h5')

# Make predictions on the training, validation, and test sets
tscv = TimeSeriesSplit(n_splits=3)
train_predictions, val_predictions, test_predictions = [], [], []
y_train_all, y_val_all, y_test_all = [], [], []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    train_size = int(len(X_train) * 0.8)
    X_train, X_val = X_train[:train_size], X_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=res.x[5], batch_size=res.x[4], verbose=1)
    
    train_predictions.extend(model.predict(X_train).flatten())
    val_predictions.extend(model.predict(X_val).flatten())
    test_predictions.extend(model.predict(X_test).flatten())
    
    y_train_all.extend(y_train)
    y_val_all.extend(y_val)
    y_test_all.extend(y_test)

# Denormalize the predictions and actual values
def denormalize(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

# Extract scalar values from the Series
min_val = output_feature.min().iloc[0]
max_val = output_feature.max().iloc[0]

train_predictions_denorm = denormalize(np.array(train_predictions), min_val, max_val)
val_predictions_denorm = denormalize(np.array(val_predictions), min_val, max_val)
test_predictions_denorm = denormalize(np.array(test_predictions), min_val, max_val)
y_train_denorm = denormalize(np.array(y_train_all), min_val, max_val)
y_val_denorm = denormalize(np.array(y_val_all), min_val, max_val)
y_test_denorm = denormalize(np.array(y_test_all), min_val, max_val)

# Ensure predictions are non-negative
train_predictions_denorm = np.maximum(train_predictions_denorm, 0)
val_predictions_denorm = np.maximum(val_predictions_denorm, 0)
test_predictions_denorm = np.maximum(test_predictions_denorm, 0)

# Create DataFrames to compare predictions with actual values
train_results = pd.DataFrame(data={'Train Predictions': train_predictions_denorm.flatten(), 'Actuals': y_train_denorm.flatten()})
val_results = pd.DataFrame(data={'Val Predictions': val_predictions_denorm.flatten(), 'Actuals': y_val_denorm.flatten()})
test_results = pd.DataFrame(data={'Test Predictions': test_predictions_denorm.flatten(), 'Actuals': y_test_denorm.flatten()})

# Calculate RMSE
train_rmse = np.sqrt(mean_squared_error(train_results['Actuals'], train_results['Train Predictions']))
val_rmse = np.sqrt(mean_squared_error(val_results['Actuals'], val_results['Val Predictions']))
test_rmse = np.sqrt(mean_squared_error(test_results['Actuals'], test_results['Test Predictions']))

print(f"Train RMSE: {train_rmse}")
print(f"Validation RMSE: {val_rmse}")
print(f"Test RMSE: {test_rmse}")

# Calculate NSE
def nse(observed, predicted):
    return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

train_nse = nse(train_results['Actuals'], train_results['Train Predictions'])
val_nse = nse(val_results['Actuals'], val_results['Val Predictions'])
test_nse = nse(test_results['Actuals'], test_results['Test Predictions'])

print(f"Train NSE: {train_nse}")
print(f"Validation NSE: {val_nse}")
print(f"Test NSE: {test_nse}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(test_results['Test Predictions'][:100], label='Test Predictions')
plt.plot(test_results['Actuals'][:100], label='Actuals')
plt.title('Test Predictions vs Actuals')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()