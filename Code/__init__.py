import os
from . import dependencies
from . import utils
from . import vis
from . import Model_training
from . import Plotting
from . import Preprocessing

# Add path functions
def get_project_root():
    """Returns absolute path to project root"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_code_dir():
    return os.path.join(get_project_root(), 'Code')

def get_data_dir():
    return os.path.join(get_project_root(), 'Data')

def get_models_dir():
    return os.path.join(get_project_root(), 'Models')

# Export everything
__all__ = [
    'dependencies', 'utils', 'vis', 'Model_trianing', 'Plotting', 'Preprocessing',
    'get_project_root', 'get_code_dir', 'get_data_dir', 'get_models_dir'
]