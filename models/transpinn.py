import torch
import torch.nn as nn

from .modules import PinnEncoder, PinnEncoderLayer, PinnFullDecoder, PinnFullDecoderLayer
from .pos_encoders import get_pos_encoder
from .factory import get_activation
from .model_utils import generate_support_points, denormalize_domain, parse_domain_from_config, compute_memory_mask

def get_sdpa_context():
    if hasattr(torch.nn, 'attention') and hasattr(torch.nn.attention, 'sdpa_kernel') and hasattr(torch.nn.attention, 'SDPBackend'):
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel(SDPBackend.MATH)
    return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

# --- SDPA Configuration ---
# Force PyTorch to use Math SDPA during graph construction
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, 'enable_math_sdp'):
    torch.backends.cuda.enable_math_sdp(True)

class TransPINN(nn.Module):
    """
    Encoder-Decoder framework for PINNs (TransPINN).
    - Encoder: Self-Attention on a set of fixed 'support points' to build a global latent map.
    - Decoder: Cross-Attention to query the latent map and evaluate u(x) at any continuous coordinate.
    """
    def __init__(self, config, problem, device, **kwargs):
        super().__init__()
        
        model_cfg = config['model']
        data_cfg = config['data']

        is_2d = isinstance(data_cfg['domain'][0], list)
        in_features = 2 if is_2d else 1
        
        self.hidden_features = model_cfg['hidden_features']
        self.output_dim = model_cfg.get('output_dim', model_cfg.get('out_features', 1))
        self.num_heads = model_cfg['num_heads']

        # Domain
        domain = data_cfg.get('domain', [0.0, 1.0])
        a, b = parse_domain_from_config(domain)
        self.register_buffer('domain_a', a)
        self.register_buffer('domain_b', b)

        # Support points
        pts = generate_support_points(
            in_features, 
            model_cfg['num_support_points'], 
            model_cfg.get('support_points_distribution', 'grid'), 
            model_cfg.get('beta_penalty', 5.0)
        )
        self.register_buffer('support_points', pts)

        # Positional encoder
        self.pos_encoder = get_pos_encoder(
            model_cfg.get('pos_encoder_type', 'lff'), 
            in_features, 
            self.hidden_features, 
            model_cfg
        )

        # Get flags
        lin_norm_flag = model_cfg.get('LinNorm', False)
        residual_norm_flag = model_cfg.get('residual_norm', True)

        # Activation
        activation = get_activation(model_cfg.get('activation', 'GELU'))

        # Encoder setup
        encoder_layer = PinnEncoderLayer(
            hidden_features=self.hidden_features,
            num_heads=self.num_heads,
            dim_feedforward=self.hidden_features * 4,
            dropout=model_cfg.get('dropout', 0.1),
            activation=activation,
            LinNorm=lin_norm_flag,
            residual_norm=residual_norm_flag
        )
        self.encoder = PinnEncoder(encoder_layer, model_cfg['num_encoder_layers'], residual_norm=residual_norm_flag)

        # Decoder setup
        decoder_layer = PinnFullDecoderLayer(
            hidden_features=self.hidden_features,
            num_heads=self.num_heads,
            dim_feedforward=self.hidden_features * 4,
            dropout=model_cfg.get('dropout', 0.1),
            activation=activation,
            LinNorm=lin_norm_flag,
            residual_norm=residual_norm_flag
        )
        self.decoder = PinnFullDecoder(decoder_layer, model_cfg['num_decoder_layers'], self.domain_a, self.domain_b, residual_norm=residual_norm_flag)

        # Final head
        self.output_head = nn.Linear(self.hidden_features, self.output_dim)
        
        # Other flags
        self.is_sequence_model = True
        causal_mask_flag = model_cfg.get('causal_mask', False)
        self.mask_fn = getattr(problem, 'causal_mask_fn', None) if causal_mask_flag else None

    def forward(self, x):
        # Force PyTorch to use Math SDPA during graph construction to allow double backpropagation
        with get_sdpa_context():
            is_2d = x.dim() == 2
            if is_2d:
                x_query = x.unsqueeze(0)
            else:
                x_query = x
                
            B, N_q, _ = x_query.shape
            device = x_query.device

            # 1. Encode support points to create the latent memory
            support_points_encoded = self.pos_encoder(self.support_points.to(device))
            
            def compute_encoder_mask(mask_fn, support_points, domain_a, domain_b):
                if mask_fn is None: return None
                sup_phys = denormalize_domain(support_points, domain_a, domain_b)
                enc_mask_bool = mask_fn(sup_phys.unsqueeze(1), sup_phys.unsqueeze(2))
                mask = torch.zeros_like(enc_mask_bool, dtype=torch.float32).squeeze(0)
                return mask.masked_fill(enc_mask_bool.squeeze(0), float('-inf'))

            encoder_mask = compute_encoder_mask(self.mask_fn, self.support_points, self.domain_a, self.domain_b)
            if encoder_mask is not None:
                encoder_mask = encoder_mask.to(device)
                
            memory = self.encoder(support_points_encoded, mask=encoder_mask)
            memory = memory.expand(B, -1, -1)

            # 2. Decode using cross-attention
            memory_mask = compute_memory_mask(self.mask_fn, self.support_points, x_query, self.domain_a, self.domain_b)
            if memory_mask is not None:
                memory_mask = memory_mask.to(device)
                # Expand for multi-head attention and ensure contiguity before reshaping/viewing.
                memory_mask = memory_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).contiguous()
                memory_mask = memory_mask.view(B * self.num_heads, N_q, -1)
            latent_output = self.decoder(
                tgt=x_query, 
                memory=memory, 
                pos_encoder=self.pos_encoder,
                memory_mask=memory_mask
            )

            # 3. Final projection
            u_hat = self.output_head(latent_output)
            
            if is_2d:
                u_hat = u_hat.squeeze(0)
                
            return u_hat

def build_model(config, problem, device, **kwargs):
    """Factory function for TransPINN."""
    model = TransPINN(
        config=config,
        problem=problem,
        device=device
    ).to(device)
    return model
