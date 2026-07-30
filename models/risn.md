# Residual Integration Solver Network (RISN)

The **RISN** architecture is a physics-informed framework specifically designed to solve complex integral and integro-differential equations. As proposed in arXiv:2501.16370v3, RISN introduces strict residual connections within deep neural networks to counteract gradient vanishing/explosion caused by fractional or non-smooth integral operators.

## Mathematical Formulation
The architecture strictly follows the recursive sequence defined in Equation (2) of the reference paper. Given an input $X \in \mathbb{R}^{N \times d}$ (where $N$ is the batch size of collocation points and $d$ is the spatial dimension), the network propagates as follows:

1. **Initialization:** $A_0 = X$
2. **Residual Hidden Layers:** For continuous hidden layers, the network applies a linear transformation, followed by a non-linear activation $\sigma$ (e.g. `Tanh`), and immediately adds the input of the layer via a skip connection:
   $$ A_i = \sigma(A_{i-1} \theta^{(i)} + b^{(i)}) + A_{i-1}, \quad i = 1, 2, \dots, L-1 $$
3. **Output Layer:** A final linear projection to derive the target function value $u(X)$:
   $$ A_L = A_{L-1} \theta^{(L)} + b^{(L)} $$

*Note:* To safely inject $X$ (which has $d$ dimensions) into a $20$-neuron hidden architecture without tensor dimension conflicts, the initial layer is executed as an input-to-hidden projection. Thus, the model achieves exactly **7 hidden layers** and **20 neurons** per layer.

## Advantages over standard MLPs
- **Robust Gradient Flow:** The explicit $+ A_{i-1}$ structure maintains stable gradient flow even when interacting with oscillatory or singular kernels.
- **Improved Optimization:** Enables the `L-BFGS` optimizer to converge to significantly lower absolute errors (e.g., $10^{-6}$) than traditional PINNs.
