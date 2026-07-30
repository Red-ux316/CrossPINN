import yaml
import collections.abc
import sys
import copy
from datetime import datetime

from config.path_manager import PathManager

def _fill_defaults(config, defaults):
    """Recursively fills missing or None keys in 'config' from 'defaults'."""
    for key, value in defaults.items():
        if isinstance(value, collections.abc.Mapping):
            if key not in config or not isinstance(config.get(key), collections.abc.Mapping):
                config[key] = {}
            _fill_defaults(config[key], value)
        elif key not in config or config[key] is None:
            config[key] = value
    return config

def print_config(config):
    """Prints the configuration in a readable YAML format."""
    from utils.logger import print_success
    print_success("[CONFIG] Final Merged Configuration:")
    print("-" * 80)
    yaml.dump(config, sys.stdout, default_flow_style=False, sort_keys=False)
    print("-" * 80)

def load_config(target, overrides=None, quiet=False):
    """
    Accepts a Study instance, PathManager instance, argparse Namespace object (args), or dictionary.
    Normalizes the input into a Study instance and returns the 3-tier merged configuration.
    """        
    from utils.logger import print_success, print_info
    # pyrefly: ignore [missing-import]
    from config.study import Study
    
    if isinstance(target, Study):
        study_obj = target
    elif isinstance(target, PathManager):
        study_obj = Study(prob_id=target.prob_id, model=target.model, study_name=target.study_name, debug=target.debug, overrides=overrides)
    elif hasattr(target, 'prob_id') or hasattr(target, 'model'):
        study_obj = Study.from_args(target, overrides=overrides)
    elif isinstance(target, dict) and 'manager' in target and target['manager']:
        pm = target['manager']
        study_obj = Study(prob_id=pm.prob_id, model=pm.model, study_name=pm.study_name, debug=pm.debug, overrides=overrides or target.get('overrides'))
    elif isinstance(target, dict):
        study_obj = Study(
            prob_id=target.get('prob_id', 1),
            model=target.get('explicit_model', target.get('model', 'mlp')),
            study_name=target.get('study_name'),
            debug=target.get('debug', False),
            overrides=overrides or target.get('overrides')
        )
    else:
        raise TypeError(f"Invalid type for load_config input: {type(target)}")

    pm = study_obj.pm
    study_path = pm.study_config_path
    base_problems_path = pm.base_problems_config_path
    problem_path = pm.problem_config_path

    base_models_path = pm.base_models_config_path
    model_category_path = pm.model_category_config_path
    models_path = pm.model_config_path
    model_type_key = pm.model
    
    if not quiet:
        print_success(f"[CONFIG] Target Study/Experiment Config Path: {study_path}")
    
    # 1. Load Study Config (most specific, optional)
    study_config = {}
    if study_path and study_path.exists():
        with open(study_path, 'r') as file:
            study_config = yaml.safe_load(file) or {}
            if not quiet:
                mtime = datetime.fromtimestamp(study_path.stat().st_mtime)
                saved_at = mtime.strftime("%d/%m %H:%M:%S")
                print_success(f"[CONFIG] Found and loaded Study Config: {study_path} (saved {saved_at})")
    else:
        if not quiet:
            print_info(f"[CONFIG] Study Config not found locally. Using defaults.")

    # 2. Load Problem Config (Base + Specific)
    base_problem_config = {}
    if base_problems_path.exists():
        with open(base_problems_path, 'r') as file:
            base_problem_config = yaml.safe_load(file) or {}
            if not quiet:
                print_success(f"[CONFIG] Loaded global base problems config: {base_problems_path.name}")

    if not problem_path.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Base problem config not found at {problem_path}")
    with open(problem_path, 'r') as file:
        specific_problem_config = yaml.safe_load(file) or {}
        if not quiet:
            print_success(f"[CONFIG] Loaded base problem config: {problem_path.name}")

    merged_problem_config = copy.deepcopy(specific_problem_config)
    _fill_defaults(merged_problem_config, base_problem_config)

    # 3. Load Models Config (Global models.yaml -> Category yaml -> Specific model.yaml)
    global_models_config = {}
    if base_models_path.exists():
        with open(base_models_path, 'r') as file:
            global_models_config = yaml.safe_load(file) or {}
            if not quiet:
                print_success(f"[CONFIG] Loaded global base models config: {base_models_path.name}")

    category_models_config = {}
    if model_category_path.exists():
        with open(model_category_path, 'r') as file:
            category_models_config = yaml.safe_load(file) or {}
            if not quiet:
                print_success(f"[CONFIG] Loaded model category config: {model_category_path.name}")

    if not models_path.exists():
        raise FileNotFoundError(f"CRITICAL ERROR: Model config not found at {models_path}")
    with open(models_path, 'r') as file:
        specific_model_config = yaml.safe_load(file) or {}
        if not quiet:
            print_success(f"[CONFIG] Loaded model specific config: {models_path.name}")

    merged_model_config = copy.deepcopy(specific_model_config)
    _fill_defaults(merged_model_config, category_models_config)
    _fill_defaults(merged_model_config, global_models_config)
                
    # 4. Merge with precedence: Study > Problem > Model
    merged_config = copy.deepcopy(study_config)
    _fill_defaults(merged_config, merged_problem_config)
    merged_config = _fill_defaults(merged_config, merged_model_config)
    
    # 5. Finalize essential keys
    if 'model' not in merged_config:
        merged_config['model'] = {}
    merged_config['model']['type'] = specific_model_config.get('model', {}).get('type', model_type_key.upper())
        
    if 'experiment' not in merged_config:
        merged_config['experiment'] = {}
    merged_config['experiment']['name'] = study_path.stem

    # 6. Automatically apply CLI / Study overrides if present
    all_overrides = study_obj.overrides or overrides
    if all_overrides:
        merged_config = apply_cli_overrides(merged_config, all_overrides, quiet=quiet)

    if 'data' in merged_config:
        if 'num_quadrature_points' in merged_config['data']:
            merged_config['data']['num_quadrature_nodes'] = merged_config['data']['num_quadrature_points']
        elif 'num_quadrature_nodes' in merged_config['data']:
            merged_config['data']['num_quadrature_points'] = merged_config['data']['num_quadrature_nodes']

    # 7. Automatically apply debug overrides if debug mode is active
    if study_obj.debug:
        merged_config = apply_debug_overrides(merged_config, study_obj)
        
    if not quiet:
        print_config(merged_config)
        
    return merged_config

def _parse_value(value_str):
    """Tries to convert a string value to float, int, bool, or None."""
    val_lower = value_str.lower()
    if val_lower == 'true':
        return True
    if val_lower == 'false':
        return False
    if val_lower in ['none', 'null']:
        return None
    try:
        if '.' in value_str or 'e' in val_lower:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return value_str

def _preprocess_overrides(overrides):
    """Normalizes overrides into a flat list of strings ('key=val')."""
    flat = []
    for item in overrides:
        if isinstance(item, dict):
            for k, v in item.items():
                flat.append(f"{k}={v}")
        elif isinstance(item, (list, tuple)):
            flat.extend(_preprocess_overrides(item))
        elif isinstance(item, str):
            flat.append(item)
    return flat

def _find_section_for_key(config, target_key):
    """Dynamically locates which section ('model', 'training', 'data', 'experiment') contains target_key."""
    if target_key in ['num_quadrature_points', 'num_quadrature_nodes']:
        return 'data'
    for section in ['model', 'training', 'data', 'experiment']:
        if section in config and isinstance(config[section], collections.abc.Mapping) and target_key in config[section]:
            return section
    return 'model'

def apply_cli_overrides(config, overrides, quiet=False):
    """
    Applies command-line overrides to the configuration dictionary.
    Supports both 'section.key=value' and un-namespaced 'key=value'.
    """
    from utils.logger import print_info, print_error
    if not overrides:
        return config

    overrides = _preprocess_overrides(overrides)

    if not quiet:
        print_info("\n[CONFIG] Applying CLI overrides:")
    for override in overrides:
        if '=' not in override:
            if not quiet:
                print_error(f"  - Invalid override format: '{override}'. Skipping. Use 'key=value'.")
            continue
        
        key_path, value_str = override.split('=', 1)
        value = _parse_value(value_str)

        if '.' not in key_path:
            section = _find_section_for_key(config, key_path)
            keys = [section, key_path]
        else:
            keys = key_path.split('.')

        d = config
        for key in keys[:-1]:
            if key not in d or not isinstance(d[key], collections.abc.Mapping):
                d[key] = {}
            d = d[key]

        target_k = keys[-1]
        original_value = d.get(target_k)
        full_path_str = '.'.join(keys)
        if not quiet:
            print_info(f"  - Overriding '{full_path_str}': {original_value} -> {value}")
        d[target_k] = value

    return config

def apply_debug_overrides(config, args):
    if getattr(args, 'debug', False):
        if 'training' in config:
            config['training']['epochs'] = 5
            config['training']['warmup_epochs'] = 0
            if 'warm_start_unfreeze_epoch' in config['training']:
                config['training']['warm_start_unfreeze_epoch'] = 2

        if 'data' in config:
            config['data']['num_collocation_points_adam'] = 100
            config['data']['num_collocation_points_lbfgs'] = 200
            config['data']['num_quadrature_nodes'] = 10
            if 'num_boundary_points_adam' in config['data']:
                config['data']['num_boundary_points_adam'] = 10

        if 'model' in config:
            if 'hidden_features' in config['model']: config['model']['hidden_features'] = 4
            if 'hidden_layers' in config['model']: config['model']['hidden_layers'] = 1
            if 'num_support_points' in config['model']: config['model']['num_support_points'] = 5
            if 'num_heads' in config['model']: config['model']['num_heads'] = 1
            if 'num_decoder_layers' in config['model']: config['model']['num_decoder_layers'] = 1
            if 'num_encoder_layers' in config['model']: config['model']['num_encoder_layers'] = 1
    return config

def add_config_arguments(parser):
    """Adds --prob_id, --model, and --study_name arguments to an argparse parser."""
    parser.add_argument('--prob_id', type=str, required=True, help='Problem ID (e.g., 1, 43, 33)')
    parser.add_argument('--model', '--method', dest='model', type=str, required=True, help='Model type (e.g., tlp, mlp, risn)')
    parser.add_argument('--study_name', type=str, default=None, help='Optional study name to override default file and experiment names.')

def get_cli_parser(description="Default MLOps script parser."):
    """Creates and returns a standard CLI parser with common MLOps arguments."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    add_config_arguments(parser)
    parser.add_argument('--debug', action='store_true', help='Run in a lightweight debug mode.')
    parser.add_argument('--reset', action='store_true', help='Clear previous artifacts and start fresh.')
    parser.add_argument('--no_wandb', dest='wandb', action='store_false', help='Disable Weights & Biases logging (enabled by default).')
    return parser

