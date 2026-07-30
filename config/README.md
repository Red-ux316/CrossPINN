# Config & Path Resolution Directory

This directory contains the central path management and configuration engine for the CrossPINN framework.

## Core Modules
- `study.py`: Contains the `Study` class for declarative experiment definitions and programmatic routines (`study.tune()`, `study.train()`, `study.evaluate()`).
- `path_manager.py`: Contains the `PathManager` class, encapsulating directory resolution (`data_root`, `weights_dir`, `plots_dir`, `logs_dir`, `tuning_dir`, model configs, problem configs, study configs).
- `config.py`: Contains `load_config(target)` and CLI argument parsers.

## Multi-Tier Configuration Hierarchy
1. **Study / Experiment Config** (`CrossPINN/equations/configs/problem_<ID>/<study_name>[_debug].yaml`): Specific execution configuration (e.g. Optuna tuned hyperparameter overrides).
2. **Problem Config**:
   - `problems.yaml` (`CrossPINN/equations/configs/problems.yaml`): Global baseline problem settings (domain `[0.0, 1.0]`, seed `42`).
   - `config_prob<ID>.yaml` (`CrossPINN/equations/configs/problem_<ID>/config_prob<ID>.yaml`): Problem-specific domain overrides.
3. **Model Config**:
   - `models.yaml` (`CrossPINN/config/yamls/models/models.yaml`): Global baseline training parameters.
   - `mlp_type.yaml` / `attention_type.yaml`: Family baseline defaults.
   - `<model>.yaml`: Model-specific hyperparameters (`tlp`, `risn`, `mlp`, `transpinn`).
