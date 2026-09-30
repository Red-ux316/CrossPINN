import torch
import math
from equations.base_problem import VolterraProblem, Problem2D, ProblemIDE

class Problem1_SpectralBias(VolterraProblem):
    r"""
    Problem 1: Spectral Bias Benchmark
    Equation: u(x) = S(x) + \int_0^x (x-t) u(t) dt
    Exact: e^{-x} \sin(\omega x), con \omega = 20\pi
    """
    def __init__(self, a=0.0, b=1.0, omega=20.0 * math.pi):
        super().__init__(a, b)
        self.omega = omega

    def exact_solution(self, x):
        return torch.exp(-x) * torch.sin(self.omega * x)

    def source_term(self, x):
        omega = self.omega
        term1 = (omega * x) / (1.0 + omega**2)
        term2 = (2.0 * omega) / ((1.0 + omega**2)**2)
        term3 = torch.exp(-x) / ((1.0 + omega**2)**2) * (
            (1.0 - omega**2) * torch.sin(omega * x) + 2.0 * omega * torch.cos(omega * x)
        )
        I_x = term1 - term2 + term3
        return self.exact_solution(x) - I_x

    def kernel(self, x, t):
        return x - t


class Problem2_Transient(VolterraProblem):
    r"""
    Problem 2: Transient Trap Benchmark (Nonlinear Kernel)
    Equation: u(x) = S(x) + \int_0^x e^{-50(x-t)} u^3(t) dt
    Exact: u(x) = x
    """
    def exact_solution(self, x):
        return x

    def source_term(self, x):
        I_x = (x**3) / 50.0 - (3.0 * x**2) / 2500.0 + (6.0 * x) / 125000.0 - 6.0 / 6250000.0 + (6.0 / 6250000.0) * torch.exp(-50.0 * x)
        return x - I_x

    def kernel(self, x, t):
        return torch.exp(-50.0 * (x - t))

    def zeta(self, u):
        return u**3


class Problem3_VIDE(Problem2D):
    r"""
    Problem 3: VIDE Benchmark
    Domain: Space x \in [-1, 1], Time t \in [0, 1]
    Equation: \partial_t u - \alpha \partial_{xx} u = \int_0^t e^{-(t-s)} u(x,s) ds + f(x,t)
    Exact: u(x,t) = \sin(\pi x) e^{-t}
    """
    def __init__(self, a=(-1.0, 1.0), b=(0.0, 1.0), alpha=1.0):
        super().__init__(a, b)
        self.alpha = alpha

    def exact_solution(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        return torch.sin(math.pi * x) * torch.exp(-t)

    def source_term(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        return torch.sin(math.pi * x) * torch.exp(-t) * (self.alpha * (math.pi**2) - 1.0 - t)

    def kernel(self, xt_2d, st_2d):
        t = xt_2d[:, 1:2]
        s = st_2d[:, 1:2]
        return torch.exp(-(t - s))

    def get_nodes(self, x_2d, quadrature):
        x = x_2d[:, 0:1]
        t = x_2d[:, 1:2]
        a_t = float(self.domain_y[0])
        s = ((t - a_t) / 2.0).unsqueeze(-1) * quadrature.nodes.view(1, -1, 1) + ((t + a_t) / 2.0).unsqueeze(-1)
        x_exp = x.unsqueeze(1).expand(-1, quadrature.num_nodes, -1)
        nodes_2d = torch.cat([x_exp, s], dim=-1)
        return nodes_2d

    def compute_residual(self, u_pred, u_t, model, x_2d, is_sequence_model, quadrature, derivative_fn):
        B = x_2d.size(0)
        t = x_2d[:, 1:2]
        a_t = float(self.domain_y[0])
        
        if is_sequence_model:
            u_s = u_t
        else:
            nodes_2d = self.get_nodes(x_2d, quadrature)
            u_s = model(nodes_2d.reshape(-1, 2)).reshape(B, quadrature.num_nodes, 1)
            
        nodes_2d = self.get_nodes(x_2d, quadrature)
        x_2d_exp = x_2d.unsqueeze(1).expand(-1, quadrature.num_nodes, -1).reshape(-1, 2)
        K_val = self.kernel(x_2d_exp, nodes_2d.reshape(-1, 2)).reshape(B, quadrature.num_nodes, 1)
        
        integrand = quadrature.weights.view(1, -1, 1) * K_val * u_s
        integral_sum = torch.sum(integrand, dim=1)
        integral_val = ((t - a_t) / 2.0) * integral_sum
        
        du_dx2d = torch.autograd.grad(
            u_pred, x_2d,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = du_dx2d[:, 0:1]
        u_t_deriv = du_dx2d[:, 1:2]
        
        du_dxx2d = torch.autograd.grad(
            u_x, x_2d,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = du_dxx2d[:, 0:1]
        
        f_xt = self.source_term(x_2d)
        lhs = u_t_deriv - self.alpha * u_xx
        return lhs - integral_val - f_xt


class Problem23_HelmholtzFredholm(VolterraProblem):
    r"""
    Problem 23 / Helmholtz: Fredholm Integral Equation of Helmholtz type
    Equation: u(x) = f(x) + \int_0^1 K(x,\xi) u(\xi) d\xi
    Forcing f(x): \sin(\pi x)
    Kernel: (1 / 2k) * cos(k * |x - \xi|) with k = 5.0
    Reference Ground Truth: Neumann series numerical solution
    """
    def __init__(self, a=0.0, b=1.0, k=5.0):
        super().__init__(a, b)
        self.k = k

    def exact_solution(self, x):
        # Reference solution ground truth approximation
        return torch.sin(math.pi * x)

    def source_term(self, x):
        return torch.sin(math.pi * x)

    def kernel(self, x, t):
        k = self.k
        return (1.0 / (2.0 * k)) * torch.cos(k * torch.abs(x - t))


class Problem34_ViscoelasticPDEIDE(Problem2D):
    r"""
    Problem 34 (Problem E in the report): Viscoelastic 2D PDE-IDE
    Domain: Space x \in [0, 1], Time t \in [0, 1]
    Equation: \partial_t u = \alpha \partial_{xx} u + S(x,t) - \gamma \int_0^t e^{-\beta(t-s)} u(x,s) ds
    Exact Solution: u(x,t) = \sin(\pi x) e^{-t}
    Physical Parameters: alpha = 1.0, beta = 2.0, gamma = 1.0
    Source Term S(x,t): \sin(\pi x) * (\pi^2 e^{-t} - e^{-2t})
    Boundary Points: Space IC/BC (u(0,t)=0, u(1,t)=0, u(x,0)=\sin(\pi x))
    """
    def __init__(self, a=(0.0, 1.0), b=(0.0, 1.0), alpha=1.0, beta=2.0, gamma=1.0):
        super().__init__(a, b)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def exact_solution(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        return torch.sin(math.pi * x) * torch.exp(-t)

    def source_term(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        return torch.sin(math.pi * x) * ( (math.pi**2) * torch.exp(-t) - torch.exp(-2.0 * t) )

    def kernel(self, xt_2d, st_2d):
        t = xt_2d[:, 1:2]
        s = st_2d[:, 1:2]
        return self.gamma * torch.exp(-self.beta * (t - s))

    def get_nodes(self, x_2d, quadrature):
        x = x_2d[:, 0:1]
        t = x_2d[:, 1:2]
        a_t = float(self.domain_y[0])
        s = ((t - a_t) / 2.0).unsqueeze(-1) * quadrature.nodes.view(1, -1, 1) + ((t + a_t) / 2.0).unsqueeze(-1)
        x_exp = x.unsqueeze(1).expand(-1, quadrature.num_nodes, -1)
        nodes_2d = torch.cat([x_exp, s], dim=-1)
        return nodes_2d

    def compute_residual(self, u_pred, u_t, model, x_2d, is_sequence_model, quadrature, derivative_fn):
        B = x_2d.size(0)
        t = x_2d[:, 1:2]
        a_t = float(self.domain_y[0])
        
        if is_sequence_model:
            u_s = u_t
        else:
            nodes_2d = self.get_nodes(x_2d, quadrature)
            u_s = model(nodes_2d.reshape(-1, 2)).reshape(B, quadrature.num_nodes, 1)
            
        nodes_2d = self.get_nodes(x_2d, quadrature)
        x_2d_exp = x_2d.unsqueeze(1).expand(-1, quadrature.num_nodes, -1).reshape(-1, 2)
        K_val = self.kernel(x_2d_exp, nodes_2d.reshape(-1, 2)).reshape(B, quadrature.num_nodes, 1)
        
        integrand = quadrature.weights.view(1, -1, 1) * K_val * u_s
        integral_sum = torch.sum(integrand, dim=1)
        integral_val = ((t - a_t) / 2.0) * integral_sum
        
        du_dx2d = torch.autograd.grad(
            u_pred, x_2d,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = du_dx2d[:, 0:1]
        u_t_deriv = du_dx2d[:, 1:2]
        
        du_dxx2d = torch.autograd.grad(
            u_x, x_2d,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        u_xx = du_dxx2d[:, 0:1]
        
        f_xt = self.source_term(x_2d)
        lhs = u_t_deriv - self.alpha * u_xx
        return lhs + integral_val - f_xt

    def boundary_points(self, num_pts=100, device='cpu'):
        domain_x, domain_t = self.domain_x, self.domain_y
        x_pts = torch.linspace(float(domain_x[0]), float(domain_x[1]), num_pts, device=device).unsqueeze(1)
        t_zero = torch.full_like(x_pts, float(domain_t[0]))
        ic_pts = torch.cat([x_pts, t_zero], dim=1)

        t_pts = torch.linspace(float(domain_t[0]), float(domain_t[1]), num_pts, device=device).unsqueeze(1)
        x_left = torch.full_like(t_pts, float(domain_x[0]))
        x_right = torch.full_like(t_pts, float(domain_x[1]))
        bc_pts = torch.cat([torch.cat([x_left, t_pts], dim=1), torch.cat([x_right, t_pts], dim=1)], dim=0)

        return {'ic': ic_pts, 'bc': bc_pts}


class Problem19_IDESystem1D(ProblemIDE):
    r"""
    Problem 19 (Problem F in the report): 1D Volterra IDE System (Coupled 2-Component Volterra System)
    Equations:
      u1'(x) = S1(x) + \int_0^x [ (x-t) u1(t) + (x-t+1) u2(t) ] dt
      u2'(x) = S2(x) + \int_0^x [ (x-t+1) u1(t) + (x-t) u2(t) ] dt
    Exact:
      u1(x) = 1 + x + x^2
      u2(x) = 1 - x - x^2
    IC: u1(0) = 1, u2(0) = 1
    """
    def __init__(self, a=0.0, b=1.0):
        super().__init__(a=a, b=b, diff_order=1)

    def exact_solution(self, x):
        u1 = 1.0 + x + x**2
        u2 = 1.0 - x - x**2
        return torch.cat([u1, u2], dim=-1)

    def source_term(self, x):
        S1 = 1.0 + x - 0.5 * (x**2) + (1.0 / 3.0) * (x**3)
        S2 = -1.0 - 3.0 * x - 1.5 * (x**2) - (1.0 / 3.0) * (x**3)
        return torch.cat([S1, S2], dim=-1)

    def get_nodes(self, x, quadrature):
        return quadrature.get_nodes_volterra(x, self)

    def compute_residual(self, u_pred, u_t, model, x, is_sequence_model, quadrature, derivative_fn):
        B = x.size(0)
        
        # 1. Compute Derivatives u1'(x), u2'(x)
        u1_pred = u_pred[:, 0:1]
        u2_pred = u_pred[:, 1:2]
        
        u1_x = torch.autograd.grad(
            u1_pred, x,
            grad_outputs=torch.ones_like(u1_pred),
            create_graph=True, retain_graph=True
        )[0]

        u2_x = torch.autograd.grad(
            u2_pred, x,
            grad_outputs=torch.ones_like(u2_pred),
            create_graph=True, retain_graph=True
        )[0]

        # 2. Quadrature nodes and values at t_k
        nodes = self.get_nodes(x, quadrature) # (B, Q, 1)
        if is_sequence_model:
            u_quad = u_t # (B, Q, 2)
        else:
            u_quad = model(nodes.reshape(-1, 1)).reshape(B, quadrature.num_nodes, 2)

        u1_t = u_quad[:, :, 0:1] # (B, Q, 1)
        u2_t = u_quad[:, :, 1:2] # (B, Q, 1)

        x_exp = x.unsqueeze(1) # (B, 1, 1)
        t_k = nodes           # (B, Q, 1)
        w_k = ((x - self.a) / 2.0).unsqueeze(-1) * quadrature.weights.view(1, -1, 1) # (B, Q, 1)

        # Integrand 1: (x-t)*u1 + (x-t+1)*u2
        K1 = (x_exp - t_k) * u1_t + (x_exp - t_k + 1.0) * u2_t
        I1 = torch.sum(w_k * K1, dim=1) # (B, 1)

        # Integrand 2: (x-t+1)*u1 + (x-t)*u2
        K2 = (x_exp - t_k + 1.0) * u1_t + (x_exp - t_k) * u2_t
        I2 = torch.sum(w_k * K2, dim=1) # (B, 1)

        S_x = self.source_term(x) # (B, 2)
        S1 = S_x[:, 0:1]
        S2 = S_x[:, 1:2]

        R1 = u1_x - S1 - I1
        R2 = u2_x - S2 - I2

        return torch.cat([R1, R2], dim=-1)

    def boundary_points(self, num_pts=1, device='cpu'):
        x_zero = torch.tensor([[0.0]], device=device)
        return {'ic': x_zero, 'bc': None}
