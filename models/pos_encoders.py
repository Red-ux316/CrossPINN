import torch
import torch.nn as nn

# --- Positional Encoders ---

class FourierEmbedding(nn.Module):
    """Random Fourier Features mapping for mitigating Spectral Bias."""
    def __init__(self, in_features, hidden_features, sigma=1.0):
        super().__init__()
        self.out_features = hidden_features
        self.half_dim = hidden_features // 2
        B = torch.randn(in_features, self.half_dim) * sigma
        self.register_buffer('B', B)
        self.linear = nn.Linear(self.half_dim * 2, hidden_features) if hidden_features % 2 != 0 else nn.Identity()

    def forward(self, x):
        x_proj = (2.0 * torch.pi) * torch.matmul(x, self.B)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return self.linear(out)

class RffEncoder(FourierEmbedding):
    """Wrapper for FourierEmbedding (Random Fourier Features)."""
    def __init__(self, in_features, hidden_features, config):
        sigma = config.get('rff_sigma', 1.0)
        super().__init__(in_features, hidden_features, sigma)

class LffEncoder(nn.Module):
    """Learnable Fourier Features (LFF) Encoder."""
    def __init__(self, in_features, hidden_features, config):
        super().__init__()
        assert hidden_features % 2 == 0, "LFF hidden_features must be even"
        self.B = nn.Parameter(torch.randn(in_features, hidden_features // 2))

    def forward(self, x):
        x_proj = 2.0 * torch.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MlpEncoder(nn.Module):
    """Classic MLP Encoder with GELU activation."""
    def __init__(self, in_features, hidden_features, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, hidden_features)
        )
    def forward(self, x):
        return self.net(x)

class SirenEncoder(nn.Module):
    """Sinusoidal MLP (SIREN) Encoder."""
    def __init__(self, in_features, hidden_features, config):
        super().__init__()
        self.omega = config.get('siren_omega', 30.0)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, hidden_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear1.weight.uniform_(-1 / self.linear1.in_features, 1 / self.linear1.in_features)
            self.linear2.weight.uniform_(-torch.sqrt(torch.tensor(6.0 / self.linear2.in_features)) / self.omega, torch.sqrt(torch.tensor(6.0 / self.linear2.in_features)) / self.omega)

    def forward(self, x):
        x = torch.sin(self.omega * self.linear1(x))
        return self.linear2(x)

class RbfEncoder(nn.Module):
    """Radial Basis Function (RBF) Encoder."""
    def __init__(self, in_features, hidden_features, config):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(hidden_features, in_features))
        self.scales = nn.Parameter(torch.ones(hidden_features))

    def forward(self, x):
        x_ = x.unsqueeze(2)
        c_ = self.centers.unsqueeze(0).unsqueeze(0)
        dist_sq = ((x_ - c_)**2).sum(-1)
        return torch.exp(-dist_sq / (self.scales.abs().square() + 1e-8))

def get_pos_encoder(encoder_type, in_features, hidden_features, config):
    """Factory function to create a positional encoder."""
    encoders = {'rff': RffEncoder, 'lff': LffEncoder, 'mlp': MlpEncoder, 'siren': SirenEncoder, 'rbf': RbfEncoder}
    encoder_class = encoders.get(encoder_type)
    if encoder_class is None:
        raise ValueError(f"Unknown pos_encoder_type: '{encoder_type}'. Available: {list(encoders.keys())}")
    return encoder_class(in_features, hidden_features, config)