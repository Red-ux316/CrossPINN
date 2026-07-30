"""
Unified Base Problem Hierarchy for PINN Equations Subsystem.
Provides abstract BaseProblem and standard category base classes (Volterra, Fredholm,
Volterra-Fredholm, Abel, IDE, 2D) to eliminate code duplication across all problem definitions.
"""

import torch

class BaseProblem:
    """Unified Abstract Base Class for Integral, Differential, and VIDE Problems."""
    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b

    def exact_solution(self, x):
        raise NotImplementedError

    def source_term(self, x):
        raise NotImplementedError

    def kernel(self, x, t):
        raise NotImplementedError

    def zeta(self, u):
        """Nonlinear function applied to u inside the integral. Defaults to identity (u)."""
        return u

    def causal_mask_fn(self, sup, q):
        """Causal attention mask. Defaults to None (no masking)."""
        return None

    def get_nodes(self, x, quadrature):
        raise NotImplementedError

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        raise NotImplementedError


class VolterraProblem(BaseProblem):
    """Base class for 1D Volterra Integral Equations of 2nd kind."""
    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra(x, self)

    def causal_mask_fn(self, sup, q):
        return (sup[..., 0] > q[..., 0]) | (sup[..., 0] < self.a)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 1.0) * u_pred
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_volterra(u_t, x, self)
        else:
            integral_val = quadrature.integrate_volterra(model, x, self)
        return lhs - S_x - integral_val


class FredholmProblem(BaseProblem):
    """Base class for 1D Fredholm Integral Equations of 2nd kind."""
    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_fredholm(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 1.0) * u_pred
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_fredholm(u_t, x, self)
        else:
            integral_val = quadrature.integrate_fredholm(model, x, self)
        return lhs - S_x - integral_val


class VolterraFredholmProblem(BaseProblem):
    """Base class for 1D Mixed Volterra-Fredholm Integral Equations."""
    def get_nodes(self, x, quadrature):
        nodes_v = quadrature.get_nodes_volterra(x, self)
        nodes_f = quadrature.get_nodes_fredholm(x, self)
        return torch.cat([nodes_v, nodes_f], dim=1)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 1.0) * u_pred
        
        if is_sequence_model:
            num_nodes = quadrature.num_nodes
            u_t_v = u_t[:, :num_nodes, :]
            u_t_f = u_t[:, num_nodes:, :]
            integral_v = quadrature.compute_integral_volterra(u_t_v, x, self)
            integral_f = quadrature.compute_integral_fredholm(u_t_f, x, self)
        else:
            integral_v = quadrature.integrate_volterra(model, x, self)
            integral_f = quadrature.integrate_fredholm(model, x, self)
        return lhs - S_x - integral_v - integral_f

    def kernel_v(self, x, t):
        raise NotImplementedError

    def kernel_f(self, x, t):
        raise NotImplementedError


class AbelProblem(BaseProblem):
    """Base class for 1D Singular Abel Integral Equations of 1st kind."""
    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 0.0) * u_pred
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_volterra(u_t, x, self)
        else:
            integral_val = quadrature.integrate_volterra(model, x, self)
        return S_x + integral_val


class ProblemIDE(BaseProblem):
    """Base class for 1D Integro-Differential Equations."""
    def __init__(self, a=0.0, b=1.0, diff_order=1):
        super().__init__(a, b)
        self.diff_order = diff_order

    def boundary_points(self, num_pts=None, device='cpu'):
        raise NotImplementedError


class Problem2D(BaseProblem):
    """Base class for 2D Spatiotemporal or Spatial Integral Equations."""
    def __init__(self, a=(0.0, 1.0), b=(0.0, 1.0)):
        super().__init__(a, b)
        self.domain_x = a
        self.domain_y = b
        self.a = a[0]
        self.b = a[1]
