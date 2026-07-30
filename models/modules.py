import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LinNormMultiheadAttention(nn.Module):
    """
    Custom MultiheadAttention that implements optional LayerNorm on Q and K projections.
    This is the core of the "AttentionNormPINN" concept to stabilize training.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, LinNorm=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.LinNorm = LinNorm
        if self.LinNorm:
            # As per documentation, apply LayerNorm to Q and K before attention scores
            self.q_norm = nn.LayerNorm(embed_dim)
            self.k_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, N_q, C = query.shape
        _    , N_k, _ = key.shape

        # 1. Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 2. Apply LinNorm if enabled (the core of the change)
        if self.LinNorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 3. Reshape for multi-head attention
        q = q.view(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 4. Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 5. Apply attention to values
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N_q, C)

        # 6. Output projection
        output = self.out_proj(context)
        return output, attn_probs

class PinnEncoderLayer(nn.Module):
    """ Standard Pre-Norm Transformer Encoder Layer. """
    def __init__(self, hidden_features, num_heads, dim_feedforward, dropout, activation, LinNorm=False, residual_norm=True):
        super().__init__()
        self.self_attn = LinNormMultiheadAttention(hidden_features, num_heads, dropout=dropout, LinNorm=LinNorm)
        
        self.linear1 = nn.Linear(hidden_features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_features)

        self.norm1 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 1. Self-attention block (Pre-Norm)
        src_norm = self.norm1(src)
        src2, _ = self.self_attn(src_norm, src_norm, src_norm, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        # 2. FFN block (Pre-Norm)
        src_norm = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src2)
        return src

class PinnEncoder(nn.Module):
    """ A stack of PinnEncoderLayers. """
    def __init__(self, encoder_layer, num_layers, residual_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.self_attn.embed_dim) if residual_norm else nn.Identity()

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.norm(output)

class PinnFullDecoderLayer(nn.Module):
    """ Standard Pre-Norm Transformer Decoder Layer with both Self- and Cross-Attention. """
    def __init__(self, hidden_features, num_heads, dim_feedforward, dropout, activation, LinNorm=False, residual_norm=True):
        super().__init__()
        self.self_attn = LinNormMultiheadAttention(hidden_features, num_heads, dropout=dropout, LinNorm=LinNorm)
        self.cross_attn = LinNormMultiheadAttention(hidden_features, num_heads, dropout=dropout, LinNorm=LinNorm)
        
        self.linear1 = nn.Linear(hidden_features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_features)

        self.norm1 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 1. Self-attention block
        tgt_norm = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # 2. Cross-attention block (memory is output of encoder, already normed)
        tgt_norm = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt_norm, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        
        # 3. FFN block
        tgt_norm = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class PinnFullDecoder(nn.Module):
    """ A stack of PinnFullDecoderLayers. """
    def __init__(self, decoder_layer, num_layers, domain_a, domain_b, residual_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.cross_attn.embed_dim) if residual_norm else nn.Identity()
        self.register_buffer('domain_a', domain_a)
        self.register_buffer('domain_b', domain_b)

    def forward(self, tgt, memory, pos_encoder, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        
        # Import here to avoid circular dependency if model_utils imports from modules
        from .model_utils import normalize_domain

        # Encode query coordinates before passing to the decoder stack
        device = tgt.device
        tgt_norm = normalize_domain(tgt, self.domain_a.to(device), self.domain_b.to(device))
        output = pos_encoder(tgt_norm)
        
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            
        return self.norm(output)

class PinnDecoderLayer(nn.Module):
    """ Simplified Pre-Norm Decoder Layer for TLP (Cross-Attention only). """
    def __init__(self, hidden_features, num_heads, dim_feedforward, dropout, activation, LinNorm=False, residual_norm=True):
        super().__init__()
        self.cross_attn = LinNormMultiheadAttention(hidden_features, num_heads, dropout=dropout, LinNorm=LinNorm)
        self.LinNorm = LinNorm
        
        self.linear1 = nn.Linear(hidden_features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_features)

        self.norm1 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden_features) if residual_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, tgt, memory, memory_mask=None, memory_key_padding_mask=None):
        # 1. Cross-attention block (Pre-Norm)
        tgt_norm = self.norm1(tgt)
        if self.LinNorm:
            mem_norm = self.norm1(memory) # Conditionally norm memory
        else:
            mem_norm = memory
        tgt2, _ = self.cross_attn(tgt_norm, mem_norm, mem_norm, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        
        # 2. FFN block (Pre-Norm)
        tgt_norm = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

class PinnDecoder(nn.Module):
    """ A stack of PinnDecoderLayers (for TLP). """
    def __init__(self, decoder_layer, num_layers, domain_a, domain_b, residual_norm=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.cross_attn.embed_dim) if residual_norm else nn.Identity()
        self.register_buffer('domain_a', domain_a)
        self.register_buffer('domain_b', domain_b)

    def forward(self, tgt, memory, pos_encoder, memory_mask=None, memory_key_padding_mask=None):
        
        # Import here to avoid circular dependency if model_utils imports from modules
        from .model_utils import normalize_domain

        # Encode query coordinates before passing to the decoder stack
        device = tgt.device
        tgt_norm = normalize_domain(tgt, self.domain_a.to(device), self.domain_b.to(device))
        output = pos_encoder(tgt_norm)
        
        for mod in self.layers:
            output = mod(output, memory, memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask)
            
        return self.norm(output)