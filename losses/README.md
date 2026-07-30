# Losses Module

This module is responsible for computing the Physics-Informed Neural Network (PINN) loss for solving integral equations.

## Components

### `pde_loss.py`
Contains the `VolterraPDELoss` class which tracks the physics-based residual.
Unlike standard differential equations, integral equations require evaluating the integral of the network's output over a specific domain. The loss is computed as the Mean Squared Error (MSE) of the residual:
$$ \mathcal{L}_{PDE} = \frac{1}{N} \sum_{i=1}^N \left| \text{LHS} - f(x_i) - \int_0^{x_i} K(x_i, t) \zeta(\hat{u}(t)) dt \right|^2 $$

**Note on Polymorphism**: `VolterraPDELoss` acts purely as a PyTorch autograd engine. The actual mathematical construction of the `LHS` and integration limits is fully delegated to the problem class via `self.problem.compute_residual(..., self.compute_derivative)`.

### `gaussian_quadrature.py`
A crucial mathematical utility for the RISN architecture. 
Since the integral $\int_0^{x_i} K(x_i, t) \zeta(\hat{u}(t)) dt$ has no analytical solution during training, it must be approximated numerically.
We utilize **Gaussian Quadrature** (Gauss-Legendre nodes and weights) for highly accurate numerical integration.
- The standard nodes $\xi_j \in [-1, 1]$ are dynamically mapped to the integration domain $[0, x_i]$ for each collocation point.
- The network is evaluated at these quadrature nodes $t_{i,j}$ in a fully vectorized and batched manner.
- This PyTorch-native implementation allows automatic differentiation to backpropagate seamlessly through the numerical integration process without breaking the computational graph.

### Fredholm Expansion
gaussian_quadrature.py now includes integrate_fredholm to map nodes strictly onto static domains $[a, b]$, allowing evaluation of dual-integration limits when dealing with Volterra-Fredholm problems.

