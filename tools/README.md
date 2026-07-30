# CrossPINN Tools & Execution Scripts

This directory contains the primary execution entry points and visualization sub-tools for running experiments, tuning hyperparameters, and plotting results in CrossPINN.

## Main Entry Points (`CrossPINN/tools/`)

### `train.py`
Main script for training a model (TLP/CrossPINN, RISN, TransPINN, MLP).
- **Usage**: `python tools/train.py --prob_id <ID> --model <MODEL_TYPE> [overrides]`
- **Functionality**: Loads hierarchical configuration via `PathManager`, constructs model and problem objects, and executes training.

### `tune.py`
Script for hyperparameter optimization using Optuna.
- **Usage**: `python tools/tune.py --prob_id <ID> --model <MODEL_TYPE> --n_trials <NUM>`
- **Functionality**: Runs Optuna trials, logs hyperparameter sweeps, saves best configs, and triggers study analysis.

### `evaluate.py`
Script for evaluating a trained PINN model against test grid / exact solutions.
- **Usage**: `python tools/evaluate.py --prob_id <ID> --model <MODEL_TYPE>`
- **Functionality**: Evaluates model checkpoints, producing L2/MAE error metrics and plots.

### `tool_utils.py`
Shared core utility functions for environment loading, optimizer/scheduler setup, Optuna plot routines, and logging.

---

## Visualization & Analysis Sub-folder (`CrossPINN/tools/visualize/`)

### `tools/visualize/plot_comparison.py`
Generates comparative benchmark reports across all models for a specific problem ID.
- **Usage**: `python tools/visualize/plot_comparison.py --prob_id <ID>`

### `tools/visualize/visualize.py`
Single-study solution visualization tool.
- **Usage**: `python tools/visualize/visualize.py --prob_id <ID> --model <MODEL>`

### `tools/visualize/analyze_study.py`
Optuna hyperparameter study analysis tool.
- **Usage**: `python tools/visualize/analyze_study.py --prob_id <ID> --model <MODEL>`