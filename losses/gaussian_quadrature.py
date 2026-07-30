import torch
import numpy as np
from scipy.special import roots_legendre

class GaussianQuadrature:
    """
    A class to perform Gaussian Quadrature integration in PyTorch.
    It maps the standard domain [-1, 1] to [a, b] dynamically based on the limits.
    For Volterra equations, the integration limits are typically [0, x].
    """
    def __init__(self, num_nodes, device='cpu'):
        self.num_nodes = num_nodes
        self.device = device
        
        # Precompute standard nodes and weights using SciPy
        nodes, weights = roots_legendre(num_nodes)
        
        # Convert to PyTorch tensors and send to device
        self.nodes = torch.tensor(nodes, dtype=torch.float32, device=device)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def _get_nodes(self, device):
        if self.nodes.device != device:
            self.nodes = self.nodes.to(device)
        return self.nodes

    def _get_weights(self, device):
        if self.weights.device != device:
            self.weights = self.weights.to(device)
        return self.weights

    def get_nodes_volterra(self, x, problem):
        """ Returns quadrature nodes for Volterra 1D. Shape [B, N_q, 1] """
        a = float(problem.a)
        nodes = self._get_nodes(x.device)
        t = ((x - a) / 2.0) * nodes.unsqueeze(0) + ((x + a) / 2.0)
        return t.unsqueeze(-1)
        
    def compute_integral_volterra(self, u_t, x, problem):
        """ Computes integral given u_t of shape [B, N_q, 1] """
        B = x.size(0)
        a = float(problem.a)
        t = self.get_nodes_volterra(x, problem).squeeze(-1) # [B, N_q]
        zeta_u = problem.zeta(u_t).reshape(B, self.num_nodes)
        weights = self._get_weights(x.device)
        
        x_expanded = x.expand(-1, self.num_nodes)
        if hasattr(problem, 'kernel_v'):
            K_val = problem.kernel_v(x_expanded, t)
        else:
            K_val = problem.kernel(x_expanded, t)
            
        integrand = weights.unsqueeze(0) * K_val * zeta_u
        integral_sum = torch.sum(integrand, dim=1, keepdim=True)
        return ((x - a) / 2.0) * integral_sum

    def integrate_volterra(self, model, x, problem):
        t = self.get_nodes_volterra(x, problem)
        u_t_flat = model(t.reshape(-1, 1))
        return self.compute_integral_volterra(u_t_flat.reshape(x.size(0), self.num_nodes, -1), x, problem)

    def get_nodes_fredholm(self, x, problem):
        """ Returns quadrature nodes for Fredholm 1D. Shape [B, N_q, 1] """
        B = x.size(0)
        a, b = float(problem.a), float(problem.b)
        nodes = self._get_nodes(x.device)
        t = ((b - a) / 2.0) * nodes.unsqueeze(0) + ((b + a) / 2.0)
        return t.expand(B, -1).unsqueeze(-1)
        
    def compute_integral_fredholm(self, u_t, x, problem):
        """ Computes integral given u_t of shape [B, N_q, 1] """
        B = x.size(0)
        a, b = float(problem.a), float(problem.b)
        t = self.get_nodes_fredholm(x, problem).squeeze(-1) # [B, N_q]
        zeta_u = problem.zeta(u_t).reshape(B, self.num_nodes)
        weights = self._get_weights(x.device)
        
        x_expanded = x.expand(-1, self.num_nodes)
        if hasattr(problem, 'kernel_f'):
            K_val = problem.kernel_f(x_expanded, t)
        else:
            K_val = problem.kernel(x_expanded, t)
            
        integrand = weights.unsqueeze(0) * K_val * zeta_u
        integral_sum = torch.sum(integrand, dim=1, keepdim=True)
        return ((problem.b - problem.a) / 2.0) * integral_sum

    def integrate_fredholm(self, model, x, problem):
        t = self.get_nodes_fredholm(x, problem)
        u_t_flat = model(t.reshape(-1, 1))
        return self.compute_integral_fredholm(u_t_flat.reshape(x.size(0), self.num_nodes, -1), x, problem)

    def get_nodes_fredholm_2d(self, x_2d, problem):
        """ Returns quadrature nodes for Fredholm 2D. Shape [B, N_q^2, 2] """
        B = x_2d.size(0)
        a_x, b_x = float(problem.domain_x[0]), float(problem.domain_x[1])
        a_y, b_y = float(problem.domain_y[0]), float(problem.domain_y[1])
        nodes = self._get_nodes(x_2d.device)
        
        t_x = ((b_x - a_x) / 2.0) * nodes + ((b_x + a_x) / 2.0)
        t_y = ((b_y - a_y) / 2.0) * nodes + ((b_y + a_y) / 2.0)
        grid_x, grid_y = torch.meshgrid(t_x, t_y, indexing='ij')
        t_2d = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        return t_2d.unsqueeze(0).expand(B, -1, -1)

    def compute_integral_fredholm_2d(self, u_t, x_2d, problem):
        """ Computes integral given u_t of shape [B, N_q^2, 1] """
        B = x_2d.size(0)
        a_x, b_x = float(problem.domain_x[0]), float(problem.domain_x[1])
        a_y, b_y = float(problem.domain_y[0]), float(problem.domain_y[1])
        weights = self._get_weights(x_2d.device)
        
        t_2d = self.get_nodes_fredholm_2d(x_2d, problem)[0] # [N_q^2, 2]
        zeta_u = problem.zeta(u_t).reshape(B, self.num_nodes**2, 1)
        
        w_x, w_y = torch.meshgrid(weights, weights, indexing='ij')
        w_2d = (w_x * w_y).reshape(-1, 1)
        w_exp = w_2d.unsqueeze(0).expand(B, -1, -1)
        
        x_exp = x_2d.unsqueeze(1).expand(-1, self.num_nodes**2, -1)
        t_exp = t_2d.unsqueeze(0).expand(B, -1, -1)
        
        K_val = problem.kernel(x_exp.reshape(-1, 2), t_exp.reshape(-1, 2)).reshape(B, self.num_nodes**2, 1)
        integrand = w_exp * K_val * zeta_u
        integral_sum = torch.sum(integrand, dim=1)
        return ((b_x - a_x) / 2.0) * ((b_y - a_y) / 2.0) * integral_sum

    def integrate_fredholm_2d(self, model, x_2d, problem):
        t = self.get_nodes_fredholm_2d(x_2d, problem)
        u_t = model(t.reshape(-1, 2))
        return self.compute_integral_fredholm_2d(u_t.reshape(x_2d.size(0), self.num_nodes**2, -1), x_2d, problem)

    def get_nodes_volterra_2d(self, x_2d, problem):
        """ Returns quadrature nodes for Volterra 2D. Shape [B, N_q^2, 2] """
        B = x_2d.size(0)
        x = x_2d[:, 0:1] 
        y = x_2d[:, 1:2]
        
        a_x = float(problem.domain_x[0])
        a_y = float(problem.domain_y[0])
        nodes = self._get_nodes(x_2d.device)
        
        t_x = ((x - a_x) / 2.0).unsqueeze(-1) * nodes.view(1, -1, 1) + ((x + a_x) / 2.0).unsqueeze(-1)
        t_y = ((y - a_y) / 2.0).unsqueeze(-1) * nodes.view(1, -1, 1) + ((y + a_y) / 2.0).unsqueeze(-1)
        
        t_x_grid = t_x.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        t_y_grid = t_y.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        return torch.cat([t_x_grid, t_y_grid], dim=-1).reshape(B, -1, 2)

    def compute_integral_volterra_2d(self, u_t, x_2d, problem):
        """ Computes integral given u_t of shape [B, N_q^2, 1] """
        B = x_2d.size(0)
        x = x_2d[:, 0:1] 
        y = x_2d[:, 1:2]
        
        a_x = float(problem.domain_x[0])
        a_y = float(problem.domain_y[0])
        weights = self._get_weights(x_2d.device)
        
        t_2d = self.get_nodes_volterra_2d(x_2d, problem) # [B, N_q^2, 2]
        zeta_u = problem.zeta(u_t).reshape(B, self.num_nodes**2, 1)
        
        w_x, w_y = torch.meshgrid(weights, weights, indexing='ij')
        w_2d = (w_x * w_y).reshape(1, -1, 1)
        
        x_exp = x_2d.unsqueeze(1).expand(-1, self.num_nodes**2, -1).reshape(-1, 2)
        K_val = problem.kernel(x_exp, t_2d.reshape(-1, 2)).reshape(B, self.num_nodes**2, 1)
        
        integrand = w_2d * K_val * zeta_u
        integral_sum = torch.sum(integrand, dim=1)
        return ((x - a_x) * (y - a_y) / 4.0) * integral_sum

    def get_nodes_spatiotemporal(self, xt_2d, problem):
        """
        Returns quadrature nodes for Spatio-temporal 2D (Fredholm in Space, Volterra in Time).
        Shape: [B, N_q^2, 2]
        """
        B = xt_2d.size(0)
        t = xt_2d[:, 1:2]
        
        a_x = float(problem.domain_x[0])
        b_x = float(problem.domain_x[1])
        a_t = float(problem.domain_y[0])
        nodes = self._get_nodes(xt_2d.device)
        
        # Space nodes (Fredholm: fixed limits)
        s_x = ((b_x - a_x) / 2.0) * nodes + ((b_x + a_x) / 2.0)
        s_x_exp = s_x.unsqueeze(0).expand(B, -1)
        
        # Time nodes (Volterra: variable upper limit t)
        s_t = ((t - a_t) / 2.0).unsqueeze(-1) * nodes.view(1, -1, 1) + ((t + a_t) / 2.0).unsqueeze(-1)
        
        # Meshgrid
        s_x_grid = s_x_exp.unsqueeze(2).expand(-1, -1, self.num_nodes)
        s_t_grid = s_t.squeeze(2).unsqueeze(1).expand(-1, self.num_nodes, -1)
        
        return torch.cat([s_x_grid.unsqueeze(-1), s_t_grid.unsqueeze(-1)], dim=-1).reshape(B, -1, 2)

    def compute_integral_spatiotemporal(self, u_t, xt_2d, problem):
        """ Computes integral given u_t of shape [B, N_q^2, 1] """
        B = xt_2d.size(0)
        t = xt_2d[:, 1:2]
        
        a_x = float(problem.domain_x[0])
        b_x = float(problem.domain_x[1])
        a_t = float(problem.domain_y[0])
        weights = self._get_weights(xt_2d.device)
        
        st_2d = self.get_nodes_spatiotemporal(xt_2d, problem) # [B, N_q^2, 2]
        zeta_u = problem.zeta(u_t).reshape(B, self.num_nodes**2, 1)
        
        w_x, w_t = torch.meshgrid(weights, weights, indexing='ij')
        w_2d = (w_x * w_t).reshape(1, -1, 1)
        
        xt_exp = xt_2d.unsqueeze(1).expand(-1, self.num_nodes**2, -1).reshape(-1, 2)
        K_val = problem.kernel(xt_exp, st_2d.reshape(-1, 2)).reshape(B, self.num_nodes**2, 1)
        
        integrand = w_2d * K_val * zeta_u
        integral_sum = torch.sum(integrand, dim=1)
        
        return ((b_x - a_x) / 2.0) * ((t - a_t) / 2.0) * integral_sum

    def integrate_spatiotemporal(self, model, xt_2d, problem):
        st = self.get_nodes_spatiotemporal(xt_2d, problem)
        u_t = model(st.reshape(-1, 2))
        return self.compute_integral_spatiotemporal(u_t.reshape(xt_2d.size(0), self.num_nodes**2, -1), xt_2d, problem)

    def integrate_volterra_2d(self, model, x_2d, problem):
        t = self.get_nodes_volterra_2d(x_2d, problem)
        u_t = model(t.reshape(-1, 2))
        return self.compute_integral_volterra_2d(u_t.reshape(x_2d.size(0), self.num_nodes**2, -1), x_2d, problem)
