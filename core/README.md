# Core Module

This directory contains the central execution loops for the Physics-Informed Neural Network (PINN).

## Files

- `trainer.py`: Defines the `Trainer` class responsible for managing the training process. It handles:
  - Dynamic generation of spatial collocation points using Latin Hypercube Sampling or uniform random generation at each epoch (Adam).
  - Hybrid optimization transitions: Instantly freezing the stochastic points into a Full Batch and swapping the optimizer to L-BFGS at a specified epoch.
  - The L-BFGS optimizer closure, including gradient zeroing, loss computation, and backpropagation.
  - Periodic evaluation of Mean Absolute Error (MAE) against the exact analytical solution on a fine uniform grid.
  - Early Stopping implementation based on MAE validation, capable of dynamically forcing an early switch to L-BFGS if Adam stagnates.
  - Checkpoint saving (`best_model.pt`) explicitly tied to historical MAE minima rather than training loss.
  - TensorBoard logging (`SummaryWriter`) for tracking the loss and MAE metrics.

- `scheduler.py`: A specialized factory module that decouples learning rate schedules from the main training loop. It parses YAML configurations to instantiate standard PyTorch schedulers (e.g. `CosineAnnealingLR`, `StepLR`) and custom schedulers like `WarmupCosineLR` tailored for Transformer stability.

- `evaluation.py`: (Optional/Planned) Contains routines for independent model evaluation on test sets and post-training analysis.

## Compliance
This module strictly follows the `AI_GUIDE.md` directives for MLOps separation, saving all checkpoints and logs to the dynamically resolved `DATA_DIR` on Google Drive, while keeping the execution logic versioned here on GitHub.
