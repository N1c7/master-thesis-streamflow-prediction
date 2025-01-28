import os
from project_paths import get_code_dir, get_data_dir, get_models_dir

from dependencies import (
    # Core data science libraries
    np,
    pd,
    plt,
    xr,
    pinv,  # numpy linear algebra
    
    # Sklearn components
    train_test_split,
    StandardScaler,
    mean_squared_error
)
input_file_path = os.path.join(get_data_dir(), 'Secchia', 'Preprocessed_data_IT.nc')
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
output_feature_smooth = output_feature

# Define the ELM model
class ELM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_weights = np.random.uniform(-1., 1., (self.input_size, self.hidden_size))
        self.biases = np.random.uniform(-1., 1., self.hidden_size)
        self.output_weights = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        self.output_weights = np.dot(pinv(H), y)

    def predict(self, X):
        H = self.sigmoid(np.dot(X, self.input_weights) + self.biases)
        predictions = np.dot(H, self.output_weights)
        return predictions

# Function to evaluate the model
def evaluate_model(hidden_size, X_train, y_train, X_test, y_test):
    elm = ELM(input_size=X_train.shape[1], hidden_size=hidden_size)
    elm.fit(X_train, y_train)
    y_pred = elm.predict(X_test)
    y_test_denorm = output_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_denorm = output_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_denorm = np.maximum(y_test_denorm, 0)  # Cap actual values at 0 after denormalizing
    y_pred_denorm = np.maximum(y_pred_denorm, 0)  # Cap predictions at 0 after denormalizing
    rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm))
    nse = 1 - (np.sum((y_test_denorm - y_pred_denorm) ** 2) / np.sum((y_test_denorm - np.mean(y_test_denorm)) ** 2))
    return rmse, nse

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=3)
hidden_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 25, 40, 50, 100, 200, 300, 400, 500]
best_rmse = float('inf')
best_nse = float('-inf')
best_hidden_size = None

for hidden_size in hidden_sizes:
    rmses = []
    nses = []
    for train_index, test_index in tscv.split(input_features):
        X_train, X_test = input_features[train_index], input_features[test_index]
        y_train, y_test = output_feature_smooth[train_index], output_feature_smooth[test_index]
        rmse, nse = evaluate_model(hidden_size, X_train, y_train, X_test, y_test)
        rmses.append(rmse)
        nses.append(nse)
    avg_rmse = np.mean(rmses)
    avg_nse = np.mean(nses)
    print(f"Hidden Size: {hidden_size}, RMSE: {avg_rmse}, NSE: {avg_nse}")
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_nse = avg_nse
        best_hidden_size = hidden_size

print(f"Best Hidden Size: {best_hidden_size}, Best RMSE: {best_rmse}, Best NSE: {best_nse}")

# Train the best model on the entire training data
elm = ELM(input_size=input_features.shape[1], hidden_size=best_hidden_size)
elm.fit(input_features, output_feature_smooth)
y_pred = elm.predict(input_features)

# Denormalize the predicted values and the actual values
y_test_denorm = output_scaler.inverse_transform(output_feature_smooth.reshape(-1, 1)).flatten()
y_pred_denorm = output_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test_denorm = np.maximum(y_test_denorm, 0)  # Cap actual values at 0 after denormalizing
y_pred_denorm = np.maximum(y_pred_denorm, 0)  # Cap predictions at 0 after denormalizing

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