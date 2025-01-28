# Core libraries - direct imports
import os
import numpy as np
import numpy.linalg as npl
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import joblib
import tensorflow as tf
import scipy.stats as stats
import scipy.interpolate as interpolate
import shapely.geometry as geometry

# Create aliases for frequently used imports
pinv = npl.pinv
YearLocator = mdates.YearLocator
DateFormatter = mdates.DateFormatter
AutoDateLocator = mdates.AutoDateLocator
pearsonr = stats.pearsonr
skew = stats.skew
kurtosis = stats.kurtosis
UnivariateSpline = interpolate.UnivariateSpline
box = geometry.box

# Sklearn components
from sklearn.model_selection import TimeSeriesSplit, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# Create aliases for sklearn metrics
sklearn_mse = mean_squared_error
mae = mean_absolute_error

# TensorFlow/Keras components
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, InputLayer, Conv1D, MaxPooling1D,
    Flatten, BatchNormalization, ReLU, Input, Concatenate,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Hyperparameter tuning
import keras_tuner as kt
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

# Local imports
from .utils import (
    nash_sutcliffe_efficiency,
    calculate_metrics,
    prepare_forecast_data,
    prepare_direct_data,
    preprocessing_data,
    calculate_weights
)
from .vis import (
    load_data,
    load_model_file,
    get_combined_predictions,
    plot_combined_model,
    plot_combined_models_comparison,
    plot_scatter_comparisons,
    plot_residual_analysis
)

# Constants and configurations
from project_paths import get_project_root
WORKING_DIRECTORY = get_project_root()

# Create a dictionary of all imports to be made available
__exports__ = {
    # Core libraries
    'os': os,
    'np': np,
    'pinv': pinv,
    'gpd': gpd,
    'pd': pd,
    'plt': plt,
    'YearLocator': YearLocator,
    'DateFormatter': DateFormatter,
    'AutoDateLocator': AutoDateLocator,
    'xr': xr,
    'joblib': joblib,
    'tf': tf,
    'pearsonr': pearsonr,
    'skew': skew,
    'kurtosis': kurtosis,
    'UnivariateSpline': UnivariateSpline,
    'box': box,
    
    # Sklearn components
    'TimeSeriesSplit': TimeSeriesSplit,
    'train_test_split': train_test_split,
    'RandomizedSearchCV': RandomizedSearchCV,
    'StandardScaler': StandardScaler,
    'mean_squared_error': mean_squared_error,
    'mean_absolute_error': mean_absolute_error,
    'r2_score': r2_score,
    'sklearn_mse': sklearn_mse,
    'mae': mae,
    
    # TensorFlow/Keras components
    'Sequential': Sequential,
    'Model': Model,
    'load_model': load_model,
    'LSTM': LSTM,
    'Dense': Dense,
    'Dropout': Dropout,
    'InputLayer': InputLayer,
    'Conv1D': Conv1D,
    'MaxPooling1D': MaxPooling1D,
    'Flatten': Flatten,
    'BatchNormalization': BatchNormalization,
    'ReLU': ReLU,
    'Input': Input,
    'Concatenate': Concatenate,
    'GlobalAveragePooling1D': GlobalAveragePooling1D,
    'Adam': Adam,
    'MeanSquaredError': MeanSquaredError,
    'RootMeanSquaredError': RootMeanSquaredError,
    'KerasRegressor': KerasRegressor,
    'ModelCheckpoint': ModelCheckpoint,
    'EarlyStopping': EarlyStopping,
    
    # Hyperparameter tuning
    'kt': kt,
    'gp_minimize': gp_minimize,
    'Real': Real,
    'Integer': Integer,
    'Categorical': Categorical,
    
    # Local imports
    'nash_sutcliffe_efficiency': nash_sutcliffe_efficiency,
    'calculate_metrics': calculate_metrics,
    'prepare_forecast_data': prepare_forecast_data,
    'prepare_direct_data': prepare_direct_data,
    'preprocessing_data': preprocessing_data,
    'calculate_weights': calculate_weights,
    'load_data': load_data,
    'load_model_file': load_model_file,
    'get_combined_predictions': get_combined_predictions,
    'plot_combined_model': plot_combined_model,
    'plot_combined_models_comparison': plot_combined_models_comparison,
    'plot_scatter_comparisons': plot_scatter_comparisons,
    'plot_residual_analysis': plot_residual_analysis,
    
    # Constants
    'WORKING_DIRECTORY': WORKING_DIRECTORY
}

# Make all exports available in the global namespace
globals().update(__exports__)

# Define what should be available for import
__all__ = list(__exports__.keys())