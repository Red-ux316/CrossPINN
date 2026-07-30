# Equations Module

This module contains the mathematical definitions for all integral and integro-differential equations solved using **CrossPINN (C-PINN)**, RISN, and baseline models. The structure is fully modularized and domain-specific to maximize scalability.

## Directory Structure

The repository organizes problems into independent sub-packages:

- **`base_problem.py`**: Unified root `BaseProblem` class and standardized category bases (`VolterraProblem`, `FredholmProblem`, `VolterraFredholmProblem`, `AbelProblem`, `ProblemIDE`, `Problem2D`).
- **`dim1/`**: 1D Integral Equations (`Volterra/`, `Fredholm/`, `VolterraFredholm/`, `Abel/`).
- **`dim2/`**: Multi-Dimensional Integral Equations (`Volterra/`, `Fredholm/`).
- **`IDE/`**: Integro-Differential Equations (`Volterra/`).
- **`Calorimeter/`**: Applied physics experiments (Particle Calorimeter simulations).
- **`Benchmark/`**: Stress-test suites for high-frequency spectral bias and stiff kernel memory decay.
- **`configs/`**: Declarative YAML problem definitions (`problems.yaml` and problem-specific configs).
- **`data_managing/`**: Spatial domain samplers (`dataset.py`).

## Polymorphic Architecture & BaseProblem Hierarchy

All problem classes inherit from `BaseProblem` (or category bases in `base_problem.py`).
To ensure models remain purely algebraic and physically agnostic, the mathematical constraints of every equation are completely decentralized into the Problem classes.

Every equation class defines:
1. `exact_solution(self, x)`: Analytical solution for $L_2$ / MAE evaluation.
2. `source_term(self, x)`: Forcing term $S(x)$ of the equation.
3. `kernel(self, x, t)`: Integration kernel $K(x, t)$.
4. `zeta(self, u)`: Non-linearity applied to $u$ inside the integral.
5. `get_nodes(self, x, quadrature)`: Quadrature node resolution.
6. `compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn)`: Polymorphic residual construction.
