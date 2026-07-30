import torch
from equations.base_problem import Problem2D

class Problem2D_Volterra1(Problem2D):
    r"""
    2D Volterra Integral Equation
    Domain: [0,1] x [0,1]
    Exact: u(x,y) = x + y
    Kernel: K(x,y,s,t) = 1
    Integral: \int_0^y \int_0^x (s+t) ds dt = (x^2 y)/2 + (x y^2)/2
    Source: S(x,y) = x + y - 0.5 * x * y * (x + y)
    """
    def __init__(self, a=(0.0, 1.0), b=(0.0, 1.0)):
        super().__init__(a, b)
        self.kappa = 1.0

    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra_2d(x, self)

    def causal_mask_fn(self, sup, q):
        a_x, b_x = self.domain_x
        a_y, b_y = self.domain_y
        mask_x = (sup[..., 0] > q[..., 0]) | (sup[..., 0] < a_x)
        mask_y = (sup[..., 1] > q[..., 1]) | (sup[..., 1] < a_y)
        return mask_x | mask_y

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        S_x = self.source_term(x)
        lhs = getattr(self, 'kappa', 1.0) * u_pred
        
        if is_sequence_model:
            integral_val = quadrature.compute_integral_volterra_2d(u_t, x, self)
        else:
            integral_val = quadrature.integrate_volterra_2d(model, x, self)
        return lhs - S_x - integral_val

    def exact_solution(self, x_2d):
        return x_2d[:, 0:1] + x_2d[:, 1:2]

    def source_term(self, x_2d):
        x = x_2d[:, 0:1]
        y = x_2d[:, 1:2]
        return x + y - 0.5 * x * y * (x + y)

    def kernel(self, x_2d, s_2d):
        return torch.ones_like(x_2d[:, 0:1])


class Problem8_Volterra2D(Problem2D_Volterra1):
    r"""
    Problem 8: Volterra 2D Benchmark (Strongly Coupled Exponential Kernel)
    Domain: [0, 1] x [0, 2]
    Exact: u(x,y) = x + y
    Kernel: K(x,y,s,t) = e^{x+y+s+t}
    Source: S(x,y) = x+y - ( (x+y-2)*e^{2x+2y} - (x-2)*e^{2x+y} - (y-2)*e^{x+2y} - 2*e^{x+y} )
    """
    def __init__(self, a=(0.0, 1.0), b=(0.0, 2.0)):
        super().__init__(a, b)

    def exact_solution(self, x_2d):
        return x_2d[:, 0:1] + x_2d[:, 1:2]

    def source_term(self, x_2d):
        x = x_2d[:, 0:1]
        y = x_2d[:, 1:2]
        exp_xy = torch.exp(x + y)
        exp_2x2y = torch.exp(2.0 * x + 2.0 * y)
        exp_2xy = torch.exp(2.0 * x + y)
        exp_x2y = torch.exp(x + 2.0 * y)

        I_xy = (x + y - 2.0) * exp_2x2y - (x - 2.0) * exp_2xy - (y - 2.0) * exp_x2y - 2.0 * exp_xy
        return (x + y) - I_xy

    def kernel(self, x_2d, s_2d):
        x = x_2d[:, 0:1]
        y = x_2d[:, 1:2]
        s = s_2d[:, 0:1]
        t = s_2d[:, 1:2]
        return torch.exp(x + y + s + t)

