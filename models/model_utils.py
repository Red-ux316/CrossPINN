import torch

def normalize_domain(x, domain_a, domain_b, target_a=0.0, target_b=1.0):
    """
    Normalizes coordinates from [domain_a, domain_b] to [target_a, target_b].
    Default target is [0, 1].
    """
    orig_range = domain_b - domain_a
    target_range = target_b - target_a
    x_norm = (x - domain_a) / orig_range
    return x_norm * target_range + target_a

def denormalize_domain(x_norm, domain_a, domain_b, target_a=0.0, target_b=1.0):
    """
    Inverse normalization: maps from [target_a, target_b] back to [domain_a, domain_b].
    Default target is [0, 1].
    """
    target_range = target_b - target_a
    orig_range = domain_b - domain_a
    x_zero_one = (x_norm - target_a) / target_range
    return x_zero_one * orig_range + domain_a

def parse_domain_from_config(domain):
    """
    Parses the domain from the configuration and returns lower and upper bound tensors.
    Handles both 1D ([min, max]) and 2D ([[xmin, xmax], [ymin, ymax]]) domains.
    """
    if isinstance(domain[0], list):
        # 2D Domain
        a = torch.tensor([d[0] for d in domain], dtype=torch.float32)
        b = torch.tensor([d[1] for d in domain], dtype=torch.float32)
    else:
        # 1D Domain
        a = torch.tensor([domain[0]], dtype=torch.float32)
        b = torch.tensor([domain[1]], dtype=torch.float32)
    return a, b

def compute_memory_mask(mask_fn, support_points, x_query, domain_a, domain_b):
    """
    Computes the additive float mask (-inf/0.0) for the Decoder's Cross-Attention.
    Includes a NaN safeguard to unmask the closest spatial point if all points are masked.
    """
    if mask_fn is None:
        return None
        
    sup_phys = denormalize_domain(support_points, domain_a, domain_b, target_a=0.0, target_b=1.0)
    sup = sup_phys.unsqueeze(1) # [1, 1, N_sup, d]
    q = x_query.unsqueeze(2)    # [B, Seq, 1, d]
    
    mask_bool = mask_fn(sup, q) # [B, Seq, N_sup]
    
    # Prevent NaN: if all support points are masked, unmask the closest one
    all_masked = mask_bool.all(dim=-1) # [B, Seq]
    if all_masked.any():
        dist = (sup - q).pow(2).sum(dim=-1) # [B, Seq, N_sup]
        closest_idx = dist.argmin(dim=-1) # [B, Seq]
        b_idx, s_idx = torch.where(all_masked)
        mask_bool[b_idx, s_idx, closest_idx[b_idx, s_idx]] = False
    
    memory_mask = torch.zeros_like(mask_bool, dtype=torch.float32)
    return memory_mask.masked_fill(mask_bool, float('-inf'))

def generate_support_points(in_features, num_support_points, distribution='grid', beta_penalty=5.0):
    """
    Generates support points in the [0, 1] normalized hypercube.
    For 1D: Uses Chebyshev nodes, uniform grid, or Beta distribution.
    For N-D: Uses Latin Hypercube Sampling (LHS).
    Returns a tensor of shape [1, num_support_points, in_features]
    """
    if in_features == 1:
        if distribution == 'beta':
            m = torch.distributions.beta.Beta(1.0, float(beta_penalty))
            pts = m.sample((num_support_points,))
            pts, _ = torch.sort(pts)
            pts = pts.view(1, num_support_points, 1)
        elif distribution == 'chebyshev':
            import numpy as np
            k = torch.arange(1, num_support_points + 1, dtype=torch.float32)
            pts = 0.5 * (1 - torch.cos((2 * k - 1) * np.pi / (2 * num_support_points)))
            pts, _ = torch.sort(pts)
            pts = pts.view(1, num_support_points, 1)
        else: # 'grid'
            pts = torch.linspace(0.0, 1.0, num_support_points).view(1, num_support_points, 1)
    else:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=in_features)
        pts = sampler.random(n=num_support_points)
        pts = torch.tensor(pts, dtype=torch.float32).unsqueeze(0)
    return pts