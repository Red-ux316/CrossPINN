import torch
import math
from equations.base_problem import VolterraProblem, Problem2D

class Calorimeter_Exp1(VolterraProblem):
    r"""
    Esperimento 1: Rivelatore Ideale (Sanity Check)
    Equazione: i(t) = S(t) + \int_0^t e^{-(t-s)} i(s) ds
    Exact: i(t) = t e^{-t}
    """
    def exact_solution(self, t):
        return t * torch.exp(-t)

    def source_term(self, t):
        return torch.exp(-t) * (t - (t**2) / 2.0)

    def kernel(self, t, s):
        return torch.exp(-(t - s))


class Calorimeter_Exp2(VolterraProblem):
    r"""
    Esperimento 2: Impurità Multiple (Stiff Memory Trap)
    Equazione: i(t) = S(t) + \int_0^t (0.9 e^{-100(t-s)} + 0.1 e^{-(t-s)}) i^2(s) ds
    Exact: i(t) = t
    """
    def exact_solution(self, t):
        return t

    def source_term(self, t):
        I_veloce = 0.9 * ((t**2)/100.0 - t/5000.0 + 1.0/500000.0 - torch.exp(-100.0 * t)/500000.0)
        I_lento = 0.1 * (t**2 - 2.0*t + 2.0 - 2.0*torch.exp(-t))
        return t - (I_veloce + I_lento)

    def kernel(self, t, s):
        return 0.9 * torch.exp(-100.0 * (t - s)) + 0.1 * torch.exp(-(t - s))

    def zeta(self, u):
        return u**2


class Calorimeter_Exp3(Problem2D):
    r"""
    Esperimento 3: Dinamica LKr Completa (VIDE 2D Spazio-Tempo)
    Domain: Spazio x \in [-1, 1], Tempo t \in [0, 1]
    Equazione: \partial_t u - D \partial_{xx} u + v_d \partial_x u = \int_0^t e^{-(t-s)} u(x,s) ds + f(x,t)
    Exact: u(x,t) = \sin(\pi x) e^{-t}
    """
    def __init__(self, a=(-1.0, 1.0), b=(0.0, 1.0), D=1.0, v_d=1.0):
        super().__init__(a, b)
        self.D = D
        self.v_d = v_d

    def exact_solution(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        return torch.sin(math.pi * x) * torch.exp(-t)

    def source_term(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        pi = math.pi
        term1 = torch.sin(pi * x) * torch.exp(-t) * (self.D * (pi**2) - 1.0 - t)
        term2 = self.v_d * pi * torch.cos(pi * x) * torch.exp(-t)
        return term1 + term2

    def kernel(self, xt_2d, st_2d):
        t = xt_2d[:, 1:2]
        s = st_2d[:, 1:2]
        return torch.exp(-(t - s))

    def get_nodes(self, x_2d, quadrature):
        B = x_2d.size(0)
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
        lhs = u_t_deriv - self.D * u_xx + self.v_d * u_x
        return lhs - integral_val - f_xt

    def causal_mask_fn(self, sup, q):
        return (sup[..., 1] > q[..., 1]) | (sup[..., 1] < self.domain_y[0])

    def boundary_points(self, num_pts=None, device='cpu'):
        if num_pts is None:
            N_ic = 100
            N_bc = 50
        else:
            N_ic = max(1, num_pts // 2)
            N_bc = max(1, num_pts // 4)

        x_ic = (torch.rand(N_ic, 1, device=device) * 2.0) - 1.0 if num_pts is not None else torch.linspace(-1.0, 1.0, N_ic, device=device).unsqueeze(1)
        t_ic = torch.zeros(N_ic, 1, device=device)
        ic_points = torch.cat([x_ic, t_ic], dim=1)
        
        t_bc_left = torch.rand(N_bc, 1, device=device) if num_pts is not None else torch.linspace(0.0, 1.0, N_bc, device=device).unsqueeze(1)
        x_left = torch.full((N_bc, 1), -1.0, device=device)
        bc_left_points = torch.cat([x_left, t_bc_left], dim=1)
        
        t_bc_right = torch.rand(N_bc, 1, device=device) if num_pts is not None else torch.linspace(0.0, 1.0, N_bc, device=device).unsqueeze(1)
        x_right = torch.full((N_bc, 1), 1.0, device=device)
        bc_right_points = torch.cat([x_right, t_bc_right], dim=1)
        
        ic_points.requires_grad_(True)
        bc_points = torch.cat([bc_left_points, bc_right_points], dim=0)
        bc_points.requires_grad_(True)
        return {'ic': ic_points, 'bc': bc_points}


class Calorimeter_Exp4(Problem2D):
    r"""
    Esperimento 4: Dinamica di Campo Non-Locale (Fredholm-Volterra IDE)
    Domain: Spazio x \in [-1, 1], Tempo t \in [0, 1]
    Equazione: \partial_t u - D \partial_{xx} u = \int_0^t \int_{-1}^1 K(x,y,t,s) u(y,s) dy ds + f(x,t)
    Exact: u(x,t) = \cos(\pi/2 x) e^{-t}
    """
    def __init__(self, a=(-1.0, 1.0), b=(0.0, 1.0), D=1.0):
        super().__init__(a, b)
        self.D = D

    def get_nodes(self, xt_2d, quadrature):
        return quadrature.get_nodes_spatiotemporal(xt_2d, self)

    def exact_solution(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        return torch.cos((math.pi / 2.0) * x) * torch.exp(-t)

    def source_term(self, xt_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        pi = math.pi
        term = torch.cos((pi / 2.0) * x) * torch.exp(-t)
        return term * (self.D * (pi**2) / 4.0 - 1.0 - t)

    def kernel(self, xt_2d, st_2d):
        x = xt_2d[:, 0:1]
        t = xt_2d[:, 1:2]
        y = st_2d[:, 0:1]
        s = st_2d[:, 1:2]
        k_space = torch.cos((math.pi / 2.0) * (x - y))
        k_time = torch.exp(-(t - s))
        return k_space * k_time

    def compute_residual(self, u_pred, u_t, model, xt_2d, is_sequence_model, quadrature, derivative_fn):
        if is_sequence_model:
            integral_val = quadrature.compute_integral_spatiotemporal(u_t, xt_2d, self)
        else:
            integral_val = quadrature.integrate_spatiotemporal(model, xt_2d, self)
            
        du_dxt = torch.autograd.grad(
            u_pred, xt_2d,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        u_x = du_dxt[:, 0:1]
        u_t_deriv = du_dxt[:, 1:2]
        
        du_dxx = torch.autograd.grad(
            u_x, xt_2d,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0][:, 0:1]
        
        f_xt = self.source_term(xt_2d)
        lhs = u_t_deriv - self.D * du_dxx
        return lhs - integral_val - f_xt

    def causal_mask_fn(self, sup, q):
        sup_t = sup[..., 1]
        q_t = q[..., 1]
        return sup_t > q_t

    def boundary_points(self, num_pts=None, device='cpu'):
        if num_pts is None:
            N_ic = 100
            N_bc = 50
        else:
            N_ic = max(1, num_pts // 2)
            N_bc = max(1, num_pts // 4)

        x_ic = (torch.rand(N_ic, 1, device=device) * 2.0) - 1.0 if num_pts is not None else torch.linspace(-1.0, 1.0, N_ic, device=device).unsqueeze(1)
        t_ic = torch.zeros(N_ic, 1, device=device)
        ic_points = torch.cat([x_ic, t_ic], dim=1)
        
        t_bc_left = torch.rand(N_bc, 1, device=device) if num_pts is not None else torch.linspace(0.0, 1.0, N_bc, device=device).unsqueeze(1)
        x_left = torch.full((N_bc, 1), -1.0, device=device)
        bc_left_points = torch.cat([x_left, t_bc_left], dim=1)
        
        t_bc_right = torch.rand(N_bc, 1, device=device) if num_pts is not None else torch.linspace(0.0, 1.0, N_bc, device=device).unsqueeze(1)
        x_right = torch.full((N_bc, 1), 1.0, device=device)
        bc_right_points = torch.cat([x_right, t_bc_right], dim=1)
        
        ic_points.requires_grad_(True)
        bc_points = torch.cat([bc_left_points, bc_right_points], dim=0)
        bc_points.requires_grad_(True)
        return {'ic': ic_points, 'bc': bc_points}
