from setuptools import setup, find_packages

setup(
    name="seasonal_streamflow_prediction",
    version="1.0",
    description="Seasonal Streamflow Prediction With Data-Driven Models",
    author="Nick van Nuland", 
    packages=find_packages(),
    py_modules=['project_paths'],
    install_requires=[
        # Core data science packages
        'numpy==1.24.4',
        'pandas==1.5.3',
        'scipy==1.10.1',
        'xarray==2023.1.0',
        'geopandas==0.13.2',
        'shapely==2.0.6',
        'netCDF4==1.6.3',
        'h5py==3.8.0',
        'Fiona==1.9.6', 
        
        # Machine Learning packages
        'scikit-learn==1.3.2',
        'tensorflow==2.9.0',
        'keras-tuner==1.4.7',
        'scikit-optimize==0.10.2',
        'protobuf<3.20,>=3.9.2',
        
        # Utilities
        'joblib==1.4.2',
        
        # Visualization
        'matplotlib==3.7.5',
    ],
    python_requires='==3.8.0'
)