import torch
from torch.utils.data import Dataset, DataLoader

class VolterraDataset(Dataset):
    """
    PyTorch Dataset for generating collocation points for 1D Volterra integral equations.
    """
    def __init__(self, a, b, num_domain_points, num_boundary_points=1, dynamic=True):
        self.a = a
        self.b = b
        self.num_domain_points = num_domain_points
        self.num_boundary_points = num_boundary_points
        self.dynamic = dynamic
        
        if not self.dynamic:
            self.x_domain = self._sample_domain()
            self.x_boundary = self._sample_boundary()

    def _sample_domain(self):
        """Uniform sampling in [a, b]."""
        return (self.b - self.a) * torch.rand(self.num_domain_points, 1) + self.a

    def _sample_boundary(self):
        """Boundary is typically x=a for Volterra equations."""
        return torch.full((self.num_boundary_points, 1), float(self.a))

    def __len__(self):
        # We define epoch length by the number of domain points
        return self.num_domain_points

    def __getitem__(self, idx):
        # PINN training typically uses full batches, this is for standard DataLoader compatibility
        if self.dynamic:
            x_i = (self.b - self.a) * torch.rand(1) + self.a
        else:
            x_i = self.x_domain[idx]
        return x_i

def get_collocation_batch(a, b, batch_size, device='cpu'):
    """
    Helper function to dynamically sample a batch of collocation points.
    Returns tensor of shape (batch_size, 1).
    """
    x = (b - a) * torch.rand(batch_size, 1, device=device) + a
    x.requires_grad_(True)
    return x

def get_boundary_batch(a, batch_size, device='cpu'):
    """
    Helper function to generate boundary points (x=a).
    """
    x_b = torch.full((batch_size, 1), float(a), device=device)
    x_b.requires_grad_(True)
    return x_b
