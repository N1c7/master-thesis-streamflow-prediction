import os
from project_paths import get_code_dir, get_data_dir, get_models_dir

from Code.dependencies import (
    # Data handling and visualization
    np,
    pd,
    plt,
    xr,
    
    # Sklearn components
    train_test_split,
    TimeSeriesSplit,
    RandomizedSearchCV,
    StandardScaler,
    mean_squared_error,
    
    # Keras/TensorFlow components
    Sequential,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    BatchNormalization,
    ReLU,
    Adam,
    KerasRegressor,
    
    # Utilities
    joblib
)

input_file_path = os.path.join(get_data_dir(), 'Iori', 'Preprocessed_data_GE.nc') 
ds = xr.open_dataset(input_file_path)

# Select input features and output feature
input_features = ds[['weighted_tp', 'weighted_t2m']].to_dataframe().dropna()
output_feature = ds['discharge'].to_dataframe().dropna()

# Align the input and output features
input_features = input_features.loc[output_feature.index]
output_feature = output_feature.loc[input_features.index]

# Normalize the input features
scaler = StandardScaler()
input_features = scaler.fit_transform(input_features)

# Normalize the output feature
output_scaler = StandardScaler()
output_feature = output_scaler.fit_transform(output_feature.values.reshape(-1, 1)).flatten()

print("Shape of input_features:", input_features.shape)
print("Shape of output_feature:", output_feature.shape)

# Reshape input data to 3D for Conv1D (samples, time steps, features)
input_features = input_features.reshape((input_features.shape[0], 1, input_features.shape[1]))

# Split the data into training, validation, and test sets (60% train, 20% validation, 20% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(input_features, output_feature, test_size=0.2, random_state=42, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, shuffle=False)  # 0.25 * 0.8 = 0.2

# Define the CNN model with 3 convolutional layers
def create_model(filters1=32, filters2=64, filters3=128, kernel_size=3, pool_size=2, dense_units1=512, dense_units2=128, learning_rate=0.001):
    model = Sequential()
    model.add(Conv1D(filters=filters1, kernel_size=kernel_size, strides=1, padding='same', input_shape=(1, input_features.shape[2])))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=pool_size, padding='same'))  
    
    model.add(Conv1D(filters=filters2, kernel_size=kernel_size, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=pool_size, padding='same')) 
    
    model.add(Conv1D(filters=filters3, kernel_size=kernel_size, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(pool_size=pool_size, padding='same')) 
    
    model.add(Flatten())
    model.add(Dense(units=dense_units1, activation='relu'))
    model.add(Dense(units=dense_units2, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

# Wrap the model using KerasRegressor
model = KerasRegressor(build_fn=create_model, epochs=50, batch_size=32, verbose=2) 

# Define the expanded hyperparameter grid
param_dist = {
    'filters1': [8, 16, 32, 64],
    'filters2': [16, 32, 64, 128],
    'filters3': [128, 256, 512],
    'kernel_size': [1, 3, 5, 7, 9],  
    'pool_size': [1, 2, 3, 4, 5],  
    'dense_units1': [128, 256, 512, 1024],
    'dense_units2': [32, 64, 128, 256],
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2, 0,1] 
}

# Perform RandomizedSearchCV with TimeSeriesSplit using the sequential backend
tscv = TimeSeriesSplit(n_splits=3)
with joblib.parallel_backend('sequential'):
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=20, cv=tscv, verbose=2, n_jobs=1) 
    random_search_result = random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found: ", random_search_result.best_params_)
print("Best score: ", random_search_result.best_score_)

# List the best hyperparameters
best_params = random_search_result.best_params_
print("Best hyperparameters:")
print(f"Filters for first Conv1D layer: {best_params['filters1']}")
print(f"Filters for second Conv1D layer: {best_params['filters2']}")
print(f"Filters for third Conv1D layer: {best_params['filters3']}")
print(f"Kernel size: {best_params['kernel_size']}")
print(f"Pool size: {best_params['pool_size']}")
print(f"Units in first Dense layer: {best_params['dense_units1']}")
print(f"Units in second Dense layer: {best_params['dense_units2']}")
print(f"Learning rate: {best_params['learning_rate']}")

# Get the best model
best_params = random_search_result.best_params_
best_model = create_model(**best_params)

# Fit the best model on the training data and capture the history
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=2) 

# Predict the values on the test set
y_pred = best_model.predict(X_test)

# Denormalize the predicted values and the actual values
y_test_denorm = output_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_denorm = output_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# Check the shapes and values
print("Shape of y_test_denorm:", y_test_denorm.shape)
print("Shape of y_pred_denorm:", y_pred_denorm.shape)
print("First 10 values of y_test_denorm:", y_test_denorm[:10])
print("First 10 values of y_pred_denorm:", y_pred_denorm[:10])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
print("RMSE: ", rmse)

# Calculate NSE
nse = 1 - (np.sum((y_test_denorm - y_pred_denorm) ** 2) / np.sum((y_test_denorm - np.mean(y_test_denorm)) ** 2))
print("NSE: ", nse)

# Plot the actual values vs the predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test_denorm, label='Actual Values')
plt.plot(y_pred_denorm, label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Streamflow')
plt.legend()
plt.show()