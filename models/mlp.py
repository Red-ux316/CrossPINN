import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Standard Multi-Layer Perceptron (Baseline PINN).
    """
    def __init__(self, in_features=1, out_features=1, hidden_features=64, hidden_layers=4, activation=nn.Tanh()):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(activation)
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(activation)
            
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
        
        # Apply proper Xavier Normal initialization for Tanh activations
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x)

def build_model(config, problem, device, **kwargs):
    """Factory function for MLP."""
    activation_name = config['model'].get('activation', 'Tanh')
    activation = torch.nn.Tanh() if activation_name == 'Tanh' else torch.nn.GELU()
    
    is_2d = isinstance(config['data']['domain'][0], list)
    in_feat = 2 if is_2d else 1
    out_feat = config['model'].get('output_dim', config['model'].get('out_features', 1))
    
    model = MLP(
        in_features=in_feat, 
        out_features=out_feat, 
        hidden_features=config['model']['hidden_features'], 
        hidden_layers=config['model'].get('hidden_layers', config['model'].get('num_blocks', 7)),
        activation=activation
    ).to(device)
    return model
