import torch
from equations.base_problem import Problem2D

class Problem2D_Fredholm1(Problem2D):
    r"""
    2D Fredholm Integral Equation
    Domain: [0,1] x [0,1]
    Exact: u(x,y) = x * y
    Kernel: K(x,y,s,t) = x * s
    Integral: \int_0^1 \int_0^1 (x*s)(s*t) ds dt = x/6
    Source: S(x,y) = x*y - x/6
    """
    def __init__(self, a=(0.0, 1.0), b=(0.0, 1.0)):
        super().__init__(a, b)
        self.kappa = 1.0

    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_fredholm_2d(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 1.0) * u_pred
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_fredholm_2d(u_t, x, self)
        else:
            integral_val = quadrature.integrate_fredholm_2d(model, x, self)
        return lhs - S_x - integral_val

    def exact_solution(self, x_2d):
        return x_2d[:, 0:1] * x_2d[:, 1:2]

    def source_term(self, x_2d):
        x = x_2d[:, 0:1]
        y = x_2d[:, 1:2]
        return x * y - x / 6.0

    def kernel(self, x_2d, s_2d):
        x = x_2d[:, 0:1]
        s = s_2d[:, 0:1]
        return x * s
