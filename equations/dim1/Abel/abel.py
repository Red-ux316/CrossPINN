import torch
from equations.base_problem import AbelProblem

class Problem5(AbelProblem):
    def exact_solution(self, x):
        return x

    def source_term(self, x):
        return (4.0 / 3.0) * (x**(1.5))

    def kernel(self, x, t):
        return -1.0 / torch.sqrt(torch.abs(x - t) + 1e-8)


class Problem6(AbelProblem):
    def exact_solution(self, x):
        return x

    def source_term(self, x):
        return (32.0 / 35.0) * (x**(3.5))

    def kernel(self, x, t):
        return -1.0 / torch.sqrt(torch.abs(x - t) + 1e-8)

    def zeta(self, u):
        return u ** 3
