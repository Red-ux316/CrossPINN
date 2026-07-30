import torch.nn as nn
import importlib

def get_activation(activation_name):
    """Factory function to get an activation function by name."""
    if activation_name == 'GELU':
        return nn.GELU()
    elif activation_name == 'Tanh':
        return nn.Tanh()
    elif activation_name == 'SiLU':
        return nn.SiLU()
    elif activation_name == 'ReLU':
        return nn.ReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")

# --- Main model factory ---

def build_model(config, problem, device, **kwargs):
    """
    Centralized factory function to instantiate models based on the YAML config.
    This function acts as a dispatcher, calling the model-specific `build_model`
    function from the corresponding module.
    """
    model_type_name = config['model']['type']
    model_type_lower = model_type_name.lower()
    
    try:
        # Dynamically import the model's module
        model_module = importlib.import_module(f"models.{model_type_lower}")
        
        # Call the build_model function within that module
        return model_module.build_model(config, problem, device, **kwargs)
        
    except ImportError:
        raise ValueError(f"Unknown model type: '{model_type_name}'. No module named 'models.{model_type_lower}.py' found.")
    except AttributeError:
        raise ValueError(f"Model module 'models.{model_type_lower}.py' does not have a 'build_model' function.")
        
    return model
