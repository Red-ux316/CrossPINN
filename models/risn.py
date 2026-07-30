import torch
import torch.nn as nn

class RISNLayer(nn.Module):
    """
    A single residual step as defined in Eq. 2 of the RISN paper:
    A_i = sigma(A_{i-1} * W + b) + A_{i-1}
    """
    def __init__(self, features, activation):
        super().__init__()
        self.linear = nn.Linear(features, features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x)) + x

class RISN(nn.Module):
    """
    Residual Integration Solver Network (RISN).
    Matches paper 2501.16370v3: exactly 7 hidden layers of 20 neurons.
    """
    def __init__(self, in_features=1, out_features=1, hidden_features=20, num_blocks=7, activation=nn.Tanh()):
        super().__init__()
        
        # Layer 1: Projection from input dimensions to hidden_features
        # We apply the activation, but no residual connection here because dimensions may not match.
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            activation
        )
        
        # Layer 2 to L (where L is num_blocks): Exactly 6 residual layers.
        # This makes a total of 1 + 6 = 7 hidden layers.
        self.hidden_layers = nn.ModuleList([
            RISNLayer(hidden_features, activation) for _ in range(num_blocks - 1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_features, out_features)
        
        # Apply proper Xavier Normal initialization for Tanh activations
        self.apply(self._init_weights)
        
        # [CRITICAL FIX]: Initialize residual blocks to exactly zero!
        # This makes the block act as a perfect identity mapping (x + Tanh(0) = x) at epoch 0,
        # preventing L-BFGS from stalling due to initial noise addition.
        for layer in self.hidden_layers:
            nn.init.constant_(layer.linear.weight, 0.0)
            if layer.linear.bias is not None:
                nn.init.constant_(layer.linear.bias, 0.0)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        return self.output_layer(out)

def build_model(config, problem, device, **kwargs):
    """Factory function for RISN."""
    activation_name = config['model'].get('activation', 'Tanh')
    activation = torch.nn.Tanh() if activation_name == 'Tanh' else torch.nn.GELU()
    
    is_2d = isinstance(config['data']['domain'][0], list)
    in_feat = 2 if is_2d else 1
    out_feat = config['model'].get('output_dim', config['model'].get('out_features', 1))
    
    model = RISN(
        in_features=in_feat, 
        out_features=out_feat, 
        hidden_features=config['model']['hidden_features'], 
        num_blocks=config['model'].get('num_blocks', 7),
        activation=activation
    ).to(device)
    return model
