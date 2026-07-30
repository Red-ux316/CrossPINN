# Data Managing Module

This module is responsible for the generation and handling of spatial collocation points and boundary points needed for training Physics-Informed Neural Networks on Volterra equations.

## Components

### `dataset.py`
Contains PyTorch utilities to sample points from the physical domain $\Omega = [a, b]$.
- **`VolterraDataset`**: A standard PyTorch `Dataset` that can optionally cache static points or generate them dynamically on the fly.
- **`get_collocation_batch`**: A fast, functional helper to generate dynamic uniform random samples directly on the target device (GPU/CPU). Since the RISN architecture relies purely on the PDE residual without external data labels, dynamic sampling during the training loop is highly effective.
- **`get_boundary_batch`**: Generates points exactly at the lower boundary $x=a$.

> [!NOTE]
> Unlike standard supervised ML, PINNs do not strictly require labeled pairs $(x, y)$. Therefore, `dataset.py` focuses entirely on spatial grid generation (collocation points) where the PDE residual is evaluated.
