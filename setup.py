from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="seasonal_streamflow_prediction",
    version="1.0",
    description="Seasonal Streamflow Prediction With Data-Driven Models",
    author="Nick van Nuland", 
    packages=find_packages(),
    py_modules=['project_paths'],
    install_requires=[
        # Core data science packages
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.5.0',
        'xarray>=0.16.0',
        'geopandas>=0.10.0',
        'shapely>=1.8.0',
        
        # Machine Learning packages
        'scikit-learn>=0.23.0',
        'tensorflow>=2.0.0',
        'keras-tuner>=1.1.0',
        'scikit-optimize>=0.8.1',
        
        # Utilities
        'joblib>=0.16.0',
        
        # Visualization
        'matplotlib>=3.2.0',
    ],
    python_requires='>=3.8'
)