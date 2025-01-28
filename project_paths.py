import os

def get_project_root():
    """Returns absolute path to project root"""
    return os.path.dirname(os.path.abspath(__file__))

def get_code_dir():
    return os.path.join(get_project_root(), 'Code')

def get_data_dir():
    return os.path.join(get_project_root(), 'Data')

def get_models_dir():
    return os.path.join(get_project_root(), 'Models')