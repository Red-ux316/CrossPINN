# Calorimeter Suite (Physics-Informed)

This module defines specialized physics-driven scenarios modeling the behavior of a Liquid Krypton (LKr) Calorimeter.

- **Problem 41 (Calorimeter Exp1 - Sanity Check)**: $i(t) = S(t) + \int_0^t e^{-(t-s)} i(s) ds$
  - Exact Solution: $i(t) = t e^{-t}$
  - Objective: Ideal detector scenario.

- **Problem 42 (Calorimeter Exp2 - Stiff Memory)**: $i(t) = S(t) + \int_0^t \left(0.9 e^{-100(t-s)} + 0.1 e^{-(t-s)}\right) i^2(s) ds$
  - Exact Solution: $i(t) = t$
  - Objective: Model the trap of multiple impurities with vastly different recombination time scales (stiff kernel).

- **Problem 43 (Calorimeter Exp3 - 2D Space-Time VIDE)**: $\partial_t u - D \partial_{xx} u + v_d \partial_x u = \int_0^t e^{-(t-s)} u(x,s) ds + f(x,t)$
  - Exact Solution: $u(x,t) = \sin(\pi x) e^{-t}$
  - Domain: $x \in [-1, 1], t \in [0, 1]$
  - Objective: Full dynamics modeling convection, diffusion, and memory. Requires strict Initial and Boundary Conditions enforcement.

- **Problem 44 (Calorimeter Exp4 - 2D Space-Time Fredholm-Volterra IDE)**: $\partial_t u - D \partial_{xx} u = \int_0^t \int_{-1}^1 \cos\left(\frac{\pi}{2}(x-y)\right) e^{-(t-s)} u(y,s) dy ds + f(x,t)$
  - Exact Solution: $u(x,t) = \cos\left(\frac{\pi}{2} x\right) e^{-t}$
  - Domain: $x \in [-1, 1], t \in [0, 1]$
  - Objective: Ultimate test case. Models a field where evolution depends instantaneously on the whole spatial domain (Fredholm) and past history (Volterra). Local operators fail completely here.
