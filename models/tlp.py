import torch
import torch.nn as nn

from .modules import PinnDecoder, PinnDecoderLayer
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

class TLP(nn.Module):
    """
    Trans Lite PINN (TLP)
    Uses a symmetric memory created from learnable values and support points,
    and a cross-attention decoder. The positional encoder is shared between
    memory creation and query encoding.
    """
    def __init__(self, config, problem, device, mlp_model=None, base_mlp_scaffold=None, mlp_config=None, **kwargs):
        super().__init__()
        
        model_cfg = config['model']
        self.data_cfg = config['data'] # Store for later use
        data_cfg = self.data_cfg
        
        is_2d = isinstance(data_cfg['domain'][0], list)
        in_features = 2 if is_2d else 1
        out_features = model_cfg.get('output_dim', model_cfg.get('out_features', 1)) 
        
        self.hidden_features = model_cfg['hidden_features']
        self.num_heads = model_cfg['num_heads']
        
        self.base_mlp_config = None # Will be populated by _setup_global_skip

        # 1. Parse Architectural Flags
        self._setup_flags(model_cfg)

        # 2. Setup Domain Buffers and Encoders
        self._setup_domain_and_encoders(data_cfg, model_cfg, in_features, out_features)

        # 3. Setup Support Points and Latent Basis (1D prior & matrix prior)
        self._setup_priors_and_warmstart(model_cfg, in_features, out_features, mlp_model)

        # 4. Setup Global Skip Connection (Base MLP)
        self._setup_global_skip(model_cfg, in_features, out_features, mlp_model, base_mlp_scaffold, mlp_config)

        # 5. Setup Cross-Attention Decoder & Output Projection Head
        self._setup_decoder(model_cfg, problem, out_features)

    def _setup_flags(self, model_cfg):
        self.use_global_skip = model_cfg.get('global_skip', False)
        self.skip_use_pretrained_mlp = model_cfg.get('skip_use_pretrained_mlp', True)
        self.skip_mlp_trainable = model_cfg.get('skip_mlp_trainable', True)
        self.learn_mem_val = model_cfg.get('learn_mem_val', False)
        self.use_pos_encoder = model_cfg.get('pos_encoder', True)

    def _setup_domain_and_encoders(self, data_cfg, model_cfg, in_features, out_features):
        domain = data_cfg.get('domain', [0.0, 1.0])
        a, b = parse_domain_from_config(domain)
        self.register_buffer('domain_a', a)
        self.register_buffer('domain_b', b)

        # Encoders setup:
        if self.use_pos_encoder:
            self.pos_encoder = get_pos_encoder(
                model_cfg.get('pos_encoder_type', 'lff'), 
                in_features, 
                self.hidden_features, 
                model_cfg
            )
            # Value encoder maps physical u scalar to latent dimension
            self.val_encoder = nn.Linear(out_features, self.hidden_features)
        else:
            # Query encoder maps x to latent dimension for cross-attention decoder
            self.pos_encoder = nn.Linear(in_features, self.hidden_features)
            # Value encoder maps joint pairs (x, u(x)) to latent dimension
            self.val_encoder = nn.Linear(in_features + out_features, self.hidden_features)

    def _setup_priors_and_warmstart(self, model_cfg, in_features, out_features, mlp_model):
        support_points = generate_support_points(
            in_features, 
            model_cfg['num_support_points'], 
            model_cfg.get('support_points_distribution', 'grid'), 
            model_cfg.get('beta_penalty', 5.0)
        )
        self.register_buffer('support_points', support_points)

        # 1D Physical prior (u_learnable)
        self.u_learnable = nn.Parameter(torch.randn(model_cfg['num_support_points'], out_features))
        
        # 2D Matrix prior (mem_val_param: [num_support_points x hidden_features])
        self.mem_val_param = nn.Parameter(torch.randn(model_cfg['num_support_points'], self.hidden_features))

        if mlp_model is not None:
            mlp_model.eval()
            with torch.no_grad():
                pts_phys = denormalize_domain(self.support_points, self.domain_a, self.domain_b, target_a=0.0, target_b=1.0)
                device_mlp = next(mlp_model.parameters()).device
                u_mlp = mlp_model(pts_phys.squeeze(0).to(device_mlp)).detach().cpu()
            
            # Freeze 1D physical prior initially
            self.u_learnable.requires_grad = False
            with torch.no_grad():
                self.u_learnable.copy_(u_mlp)
                if self.use_pos_encoder:
                    initial_mem_val = self.val_encoder(u_mlp)
                else:
                    joint_pts_u = torch.cat([self.support_points.squeeze(0), u_mlp], dim=-1)
                    initial_mem_val = self.val_encoder(joint_pts_u)
                self.mem_val_param.copy_(initial_mem_val)
            self.mem_val_param.requires_grad = False
            
            self.is_in_warmup_phase = True
        else:
            self.is_in_warmup_phase = False
            if self.learn_mem_val:
                self.u_learnable.requires_grad = False
                self.mem_val_param.requires_grad = True
            else:
                self.mem_val_param.requires_grad = False
                self.u_learnable.requires_grad = True


    def _setup_global_skip(self, model_cfg, in_features, out_features, mlp_model, base_mlp_scaffold=None, mlp_config=None):
        self.base_mlp = None
        self.base_mlp_config = None
        if not self.use_global_skip:
            return

        from models.mlp import MLP
        from .factory import get_activation

        # Priority 1: Use a pre-built scaffold if provided (for robust evaluation from a self-contained checkpoint)
        if base_mlp_scaffold is not None:
            self.base_mlp = base_mlp_scaffold
            self.base_mlp_config = mlp_config # The config used to build the scaffold
            for p in self.base_mlp.parameters():
                p.requires_grad = self.skip_mlp_trainable
        # Priority 2: Use the warm-start MLP if configured (for training)
        elif self.skip_use_pretrained_mlp and (mlp_model is not None):
            self.base_mlp = mlp_model
            self.base_mlp_config = mlp_config # The config used for the warm-start MLP
            for p in self.base_mlp.parameters():
                p.requires_grad = self.skip_mlp_trainable
        # Priority 3: Build a new MLP from scratch (for cold-start training)
        else:
            activation_name = model_cfg.get('activation', 'GELU')
            hidden_layers_mlp = model_cfg.get('hidden_layers', 4)
            activation_fn = get_activation(activation_name)
            self.base_mlp = MLP(
                in_features=in_features, 
                out_features=out_features, 
                hidden_features=self.hidden_features, 
                hidden_layers=hidden_layers_mlp,
                activation=activation_fn
            )
            # Create a config dict for this newly created MLP
            self.base_mlp_config = {
                'model': {
                    'type': 'mlp',
                    'hidden_features': self.hidden_features,
                    'hidden_layers': hidden_layers_mlp,
                    'activation': activation_name,
                    'output_dim': out_features
                },
                'data': { 'domain': self.data_cfg['domain'] }
            }
            for p in self.base_mlp.parameters():
                p.requires_grad = True

    def _setup_decoder(self, model_cfg, problem, out_features):
        lin_norm_flag = model_cfg.get('LinNorm', False)
        residual_norm_flag = model_cfg.get('residual_norm', True)
        activation = get_activation(model_cfg.get('activation', 'GELU'))
        
        decoder_layer = PinnDecoderLayer(
            hidden_features=self.hidden_features,
            num_heads=self.num_heads,
            dim_feedforward=self.hidden_features * 4,
            dropout=model_cfg.get('dropout', 0.1),
            activation=activation,
            LinNorm=lin_norm_flag,
            residual_norm=residual_norm_flag
        )
        self.decoder = PinnDecoder(
            decoder_layer, 
            model_cfg['num_decoder_layers'], 
            self.domain_a, 
            self.domain_b, 
            residual_norm=residual_norm_flag
        )

        self.output_head = nn.Linear(self.hidden_features, out_features)
        if self.use_global_skip:
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

        self.is_sequence_model = True
        causal_mask_flag = model_cfg.get('causal_mask', False)
        self.mask_fn = getattr(problem, 'causal_mask_fn', None) if causal_mask_flag else None

    def transition_to_free_basis(self):
        """
        Called by the trainer to transition from Phase 1 (fixed physical prior) 
        to Phase 2 (learnable physical prior).
        """
        if self.is_in_warmup_phase:
            if self.learn_mem_val:
                self.mem_val_param.requires_grad = True
                self.u_learnable.requires_grad = False
            else:
                self.u_learnable.requires_grad = True
                self.mem_val_param.requires_grad = False
            self.is_in_warmup_phase = False

    def forward(self, x):
        # Force PyTorch to use Math SDPA during graph construction to allow double backpropagation
        with get_sdpa_context():
            is_unbatched = x.dim() == 2
            if is_unbatched:
                x_query = x.unsqueeze(0)
            else:
                x_query = x
                
            B, N_q, _ = x_query.shape
            device = x_query.device
            
            # 1. Create memory
            if self.use_pos_encoder:
                mem_pos = self.pos_encoder(self.support_points.to(device))
                if self.learn_mem_val:
                    mem_val = self.mem_val_param.to(device)
                else:
                    mem_val = self.val_encoder(self.u_learnable.to(device))
                memory = (mem_pos + mem_val).expand(B, -1, -1)
            else:
                if self.learn_mem_val:
                    mem_val = self.mem_val_param.to(device)
                    memory = mem_val.unsqueeze(0).expand(B, -1, -1) if mem_val.dim() == 2 else mem_val.expand(B, -1, -1)
                else:
                    joint_sup_u = torch.cat([self.support_points.to(device), self.u_learnable.to(device).unsqueeze(0)], dim=-1)
                    memory = self.val_encoder(joint_sup_u).expand(B, -1, -1)

            
            # 2. Compute mask
            memory_mask = compute_memory_mask(self.mask_fn, self.support_points, x_query, self.domain_a, self.domain_b)
            if memory_mask is not None:
                memory_mask = memory_mask.to(device)
                # Expand for multi-head attention and ensure contiguity before reshaping/viewing.
                memory_mask = memory_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).contiguous()
                memory_mask = memory_mask.view(B * self.num_heads, N_q, -1)
            
            # 3. Decode
            latent_output = self.decoder(
                tgt=x_query, 
                memory=memory, 
                pos_encoder=self.pos_encoder,
                memory_mask=memory_mask
            )
            
            u_attention = self.output_head(latent_output)
            
            if self.use_global_skip:
                u_base = self.base_mlp(x_query)
                u_hat = u_base + u_attention
            else:
                u_hat = u_attention
            
            if is_unbatched:
                u_hat = u_hat.squeeze(0)
                
            return u_hat

def build_model(config, problem, device, debug=False, quiet=False, **kwargs):
    """
    Factory function for TLP. Handles warm-start by loading a pre-trained MLP if warm_start is True.
    """
    from config.config import load_config, apply_debug_overrides
    from config.path_manager import PathManager
    from models.mlp import build_model as build_mlp_model
    import argparse
    from pathlib import Path

    mlp_model = None
    mlp_config = None
    
    # Check if warm-start is enabled in the config
    use_warm_start = config.get('training', {}).get('warm_start', False)
    warm_start_study_name = config.get('training', {}).get('warm_start_study_name', None)

    if use_warm_start:
        prob_id = config['data']['problem_id']
        target_study_name = warm_start_study_name or f"MLP_prob{prob_id}"
        mlp_pm = PathManager(
            prob_id=prob_id,
            model='mlp',
            study_name=target_study_name,
            debug=debug
        )
        
        mlp_config = load_config(mlp_pm, quiet=True)

        if debug:
            mlp_config = apply_debug_overrides(mlp_config, argparse.Namespace(debug=debug))

        mlp_save_path = mlp_pm.get_weight_checkpoint_path(checkpoint_type="best")

        if not mlp_save_path.exists():
            if not quiet:
                from utils.logger import print_error
                print_error(f"[Warm Start] MLP checkpoint not found at: {mlp_save_path}")
                print_error(" -> Falling back to cold start initialization.")
            mlp_model = None
        else:
            try:
                import copy
                from models.mlp import build_model as build_mlp_model

                checkpoint = torch.load(mlp_save_path, map_location=device, weights_only=False)
                state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                
                if isinstance(checkpoint, dict) and 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                    ckpt_mlp_config = checkpoint['config']
                else:
                    # Fallback for old checkpoints without embedded config
                    ckpt_mlp_config = mlp_config
                
                # Ensure the hidden_features of the loaded MLP matches the TLP's requirements if they are different
                build_config = copy.deepcopy(ckpt_mlp_config)
                # The config from the MLP checkpoint is the source of truth for its architecture.
                build_config['data'] = config['data']

                temp_mlp_model = build_mlp_model(build_config, problem, device, quiet=True)
                temp_mlp_model.load_state_dict(state_dict)
                temp_mlp_model.eval()
                mlp_model = temp_mlp_model
                mlp_config = build_config
            except Exception as e:
                if not quiet:
                    from utils.logger import print_error
                    print_error(f"[Warm Start] Could not load MLP checkpoint at {mlp_save_path}: {e}")
                    print_error(" -> Falling back to cold start initialization.")
                mlp_model = None
    
    # Now, build the actual TLP model
    model = TLP(
        config=config,
        problem=problem,
        device=device,
        mlp_model=mlp_model,
        base_mlp_scaffold=kwargs.pop('base_mlp_scaffold', None),
        mlp_config=kwargs.pop('mlp_config', mlp_config),
    ).to(device)
    
    return model
