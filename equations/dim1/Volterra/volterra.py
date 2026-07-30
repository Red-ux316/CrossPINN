import torch
from equations.base_problem import VolterraProblem

class Problem1(VolterraProblem):
    r"""
    Volterra Integral Equation of the Second Kind (Linear):
    u(x) = S(x) + \int_0^x K(x,t) u(t) dt
    
    From RISN paper (1D Integral Equations):
    Kernel: K(x,t) = t - x
    Exact Solution: u(x) = x + exp(x)
    Source Term: S(x) = 2*exp(x) - 1 + (x^3)/6
    Interval: [0, 1]
    """
    def exact_solution(self, x):
        return x + torch.exp(x)

    def source_term(self, x):
        return 2 * torch.exp(x) - 1 + (x**3) / 6.0

    def kernel(self, x, t):
        return t - x


class Problem2(VolterraProblem):
    r"""
    Volterra Integral Equation of the Second Kind (Nonlinear):
    u(x) = S(x) + \int_0^x K(x,t) zeta(u(t)) dt
    
    From RISN paper (1D Integral Equations):
    Kernel: K(x,t) = 1
    Exact Solution: u(x) = exp(x)
    Source Term: S(x) = exp(x) - 0.5 * (exp(2x) - 1)
    Zeta(u): u^2 (Nonlinear)
    Interval: [0, 1]
    """
    def exact_solution(self, x):
        return torch.exp(x)

    def source_term(self, x):
        return torch.exp(x) - 0.5 * (torch.exp(2 * x) - 1.0)

    def kernel(self, x, t):
        return torch.ones_like(t)

    def zeta(self, u):
        return u ** 2
