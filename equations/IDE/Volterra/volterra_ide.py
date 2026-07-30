import torch
from equations.base_problem import ProblemIDE

class ProblemIDE_Volterra1(ProblemIDE):
    r"""
    1D Volterra Integro-Differential Equation
    Equation: u'(x) = S(x) + \int_0^x u(t) dt
    Exact: sin(x)
    Source: 2*cos(x) - 1
    Order: 1
    BCs: u(0) = 0
    """
    def __init__(self, a=0.0, b=1.0):
        super().__init__(a, b, diff_order=1)
        self.kappa = 1.0

    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 1.0) * derivative_fn(u_pred, x, self.diff_order)
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_volterra(u_t, x, self)
        else:
            integral_val = quadrature.integrate_volterra(model, x, self)
        return lhs - S_x - integral_val

    def exact_solution(self, x):
        return torch.sin(x)

    def source_term(self, x):
        return 2.0 * torch.cos(x) - 1.0

    def kernel(self, x, t):
        return torch.ones_like(t)

    def boundary_points(self, num_pts=None, device='cpu'):
        pts = torch.tensor([[0.0]], device=device, requires_grad=True)
        return {'ic': pts, 'bc': None}


class Problem24_VIDE_Nonlinear_1stOrder(ProblemIDE):
    r"""
    Problem 24: 1D Non-linear Volterra IDE (1st Order)
    Equation: u'(x) = S(x) + \int_0^x u^2(t) dt
    Exact: u(x) = x
    Source: 1 - x^3/3
    Order: 1
    IC: u(0) = 0
    """
    def __init__(self, a=0.0, b=1.0):
        super().__init__(a, b, diff_order=1)

    def exact_solution(self, x):
        return x

    def source_term(self, x):
        return 1.0 - (x**3) / 3.0

    def kernel(self, x, t):
        return torch.ones_like(t)

    def zeta(self, u):
        return u**2

    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = derivative_fn(u_pred, x, self.diff_order)
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_volterra(u_t, x, self)
        else:
            integral_val = quadrature.integrate_volterra(model, x, self)
        
        return lhs - S_x - integral_val

    def boundary_points(self, num_pts=None, device='cpu'):
        pts = torch.tensor([[self.a]], device=device, requires_grad=True)
        return {'ic': pts, 'bc': None}


class Problem25_VIDE_Linear_2ndOrder(ProblemIDE):
    r"""
    Problem 25: 1D Linear Volterra IDE (2nd Order)
    Equation: u''(x) = S(x) + \int_0^x (x-t)u(t) dt
    Exact: u(x) = sin(x)
    Source: -x
    Order: 2
    BCs: u(0) = 0, u(1) = sin(1) (adapted to BVP for framework compatibility)
    """
    def __init__(self, a=0.0, b=1.0):
        super().__init__(a, b, diff_order=2)

    def exact_solution(self, x):
        return torch.sin(x)

    def source_term(self, x):
        return -x

    def kernel(self, x, t):
        return x - t

    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = derivative_fn(u_pred, x, self.diff_order)
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_volterra(u_t, x, self)
        else:
            integral_val = quadrature.integrate_volterra(model, x, self)
        
        return lhs - S_x - integral_val

    def boundary_points(self, num_pts=None, device='cpu'):
        # Using Dirichlet BCs at both ends to form a Boundary Value Problem
        # u(a) = sin(a), u(b) = sin(b)
        pts = torch.tensor([[self.a], [self.b]], device=device, requires_grad=True)
        return {'ic': None, 'bc': pts}
