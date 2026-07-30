import torch
from equations.base_problem import VolterraFredholmProblem

class Problem3(VolterraFredholmProblem):
    def exact_solution(self, x):
        return x + torch.exp(x)

    def source_term(self, x):
        e_val = torch.exp(torch.tensor(1.0, device=x.device))
        return 2 * torch.exp(x) - (x / 2.0) - (7.0 / 3.0) + (x**3) / 6.0 + x * e_val

    def kernel_v(self, x, t):
        return t - x

    def kernel_f(self, x, t):
        return t - x


class Problem4(VolterraFredholmProblem):
    def exact_solution(self, x):
        return x * torch.exp(x)

    def source_term(self, x):
        return torch.exp(x) - 1.0 - x

    def kernel_v(self, x, t):
        return torch.ones_like(t)

    def kernel_f(self, x, t):
        return x.expand(-1, t.size(1)) if x.dim() > 1 else x * torch.ones_like(t)
