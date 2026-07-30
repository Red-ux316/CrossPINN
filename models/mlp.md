# Multi-Layer Perceptron (Baseline PINN)

The **MLP** is the standard feed-forward neural network utilized as the baseline for evaluating Physics-Informed Neural Networks. 

## Architectural Setup
To guarantee a mathematically fair comparison against the advanced RISN model (as prescribed in Section 4 of arXiv:2501.16370v3), the baseline MLP is configured with identical capacity:
- **Depth:** 7 hidden layers
- **Width:** 20 neurons per layer
- **Activation:** `Tanh` across all hidden layers

Unlike RISN, the MLP does not employ residual or skip connections. The forward pass is a continuous composition of linear transformations and activations:
$$ A_i = \sigma(A_{i-1} \theta^{(i)} + b^{(i)}) $$

## Limitations
In complex problems—such as fractional differential equations or Volterra systems with highly singular kernels—the deep structure of a standard MLP can lead to vanishing gradients. Without residual connections to stabilize backpropagation, it struggles to match the accuracy and stability of RISN even when trained with `L-BFGS` on identical collocation grids.
