# Benchmark Suite

This module contains pathological edge cases designed specifically to stress-test Neural Architectures and reveal standard PINN limitations (e.g. Spectral Bias, stiff memory tracking).

- **Problem 31 (Spectral Bias / High Frequency)**: $u(x) = S(x) + \int_0^x (x-t) u(t) dt$
  - Exact Solution: $u(x) = e^{-x} \sin(20\pi x)$
  - Objective: Prove the inability of standard pointwise MLPs to learn highly oscillatory target functions.

- **Problem 32 (Transient Trap / Stiff Kernel)**: $u(x) = S(x) + \int_0^x e^{-50(x-t)} u^3(t) dt$
  - Exact Solution: $u(x) = x$
  - Objective: Evaluate the model's response to an extremely sharp, stiff, and localized exponential memory decay.

- **Problem 33 (Gradient Stiffness in VIDE)**: $u''(x) = S(x) + \int_0^x u^2(t) dt$
  - Exact Solution: $u(x) = \cos(2\pi x)$
  - Boundary Conditions: $u(0) = 1$, $u(1) = 1$
  - Objective: Test the interplay between high-order derivatives and non-linear memory accumulation.
