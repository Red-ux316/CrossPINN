# Models Module

This module contains the neural network architectures utilized to approximate the unknown function $u(x)$ in integral and integro-differential equations.

## Architectures

### `tlp.py` (CrossPINN / C-PINN)
The flagship **CrossPINN** architecture. It replaces expensive self-attention sequence encoders with a learnable latent memory matrix ($N_{\text{sup}} \times H$) interlinked via Sequential Cross-Attention.
- **Positional Encoder Integration**: Shared spatial positional encoders mapping physical query coordinates $x_{\text{query}}$ to hidden latent spaces.
- **Global Skip Connection (`global_skip: true`)**: Parallel Base MLP stream ($\hat{u} = \text{MLP} + \Delta u_{\text{Attn}}$) ensuring physics prior stability across stiff loss landscapes.
- **Direct Latent Basis Learning**: Learnable memory representation bypassing heavy 1D value projections.
- **Two-Phase Warm-Start**: Initializing the MLP prior to zero-output state (`nn.init.zeros_`) for smooth optimization transitions.

### `risn.py` (Residual Integration Solver Network)
The architecture proposed in arXiv:2501.16370. Incorporates residual skip connections between hidden layers to mitigate vanishing gradients in deep networks.

### `transpinn.py` (TransPINN)
Full support-point self-attention Operator Learning framework with an Encoder-Decoder pipeline.

### `mlp.py` (Baseline MLP)
Standard feed-forward Multi-Layer Perceptron used as the primary baseline.

### `factory.py` (Factory Pattern)
Centralized dispatcher `build_model(config, problem, device)` instantiating models dynamically based on YAML configurations.

---

## Detailed Mathematical Formulation
For in-depth mathematical derivations of each architecture:
- [CrossPINN / TLP Formulation](tlp.md)
- [RISN Formulation](risn.md)
- [MLP Formulation](mlp.md)
- [TransPINN Formulation](trans_pinn.md)
