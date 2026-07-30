import torch
import torch.nn as nn
from .gaussian_quadrature import GaussianQuadrature
class VolterraPDELoss(nn.Module):
    """
    Computes the Physics-Informed residual loss for 1D/2D Integral and Integro-Differential Equations.
    """
    def __init__(self, problem, num_quadrature_nodes=100, device='cpu'):
        super().__init__()
        self.problem = problem
        self.quadrature = GaussianQuadrature(num_quadrature_nodes, device=device)
        self.mse_loss = nn.MSELoss()

    def compute_derivative(self, u, x, order):
        du_dx = u
        for _ in range(order):
            du_dx = torch.autograd.grad(
                du_dx, x, 
                grad_outputs=torch.ones_like(du_dx),
                create_graph=True,
                retain_graph=True
            )[0]
        return du_dx

    def forward(self, model, x):
        is_sequence_model = getattr(model, 'is_sequence_model', False)
        
        if is_sequence_model:
            # Generate nodes and construct sequence using polymorphism
            nodes = getattr(self.problem, 'get_nodes', lambda x, q: None)(x, self.quadrature)
                    
            if nodes is not None:
                seq = torch.cat([x.unsqueeze(1), nodes], dim=1) # [B, K+1, d]
                U = model(seq) # [B, K+1, out_dim]
                u_pred = U[:, 0, :]
                u_t = U[:, 1:, :]
            else:
                u_pred = model(x.unsqueeze(1)).squeeze(1)
                u_t = None
        else:
            u_pred = model(x)
            u_t = None

        residual = self.problem.compute_residual(
            u_pred, u_t, model, x, is_sequence_model, self.quadrature, self.compute_derivative
        )
        
        target_zero = torch.zeros_like(residual)
        loss = self.mse_loss(residual, target_zero)
        
        return loss
