# 1D Integral Equations

This module defines classes for solving classic one-dimensional Integral Equations. All problems default to the spatial domain $x \in [0, 1]$.

## Volterra Integral Equations (Second Kind)
Integral equations where the upper limit of integration varies with $x$.
- **Problem 1 (Linear)**: $u(x) = S(x) + \int_0^x (t-x) u(t) dt$. Exact Solution: $u(x) = x + e^x$.
- **Problem 2 (Nonlinear)**: $u(x) = S(x) + \int_0^x u^2(t) dt$. Exact Solution: $u(x) = e^x$.

## Volterra-Fredholm Integral Equations (Second Kind)
Mixed equations featuring both a variable boundary integral (Volterra) and a fixed boundary integral (Fredholm).
- **Problem 3 (Linear)**: $u(x) = S(x) + \int_0^x (t-x) u(t) dt + \int_0^1 (t-x) u(t) dt$. Exact Solution: $u(x) = x + e^x$.
- **Problem 4 (Nonlinear)**: $u(x) = S(x) + \int_0^x u(t) dt + \int_0^1 x u(t) dt$. Exact Solution: $u(x) = x e^x$.

## Abel Integral Equations (First Kind)
Equations of the First Kind where the unknown function $u(x)$ appears *only* inside the integral, characterized by a weakly singular algebraic kernel.
- **Problem 5 (Linear)**: $\int_0^x \frac{-1}{\sqrt{x-t}} u(t) dt + S(x) = 0$. Exact Solution: $u(x) = x$.
- **Problem 6 (Nonlinear)**: $\int_0^x \frac{-1}{\sqrt{x-t}} u^3(t) dt + S(x) = 0$. Exact Solution: $u(x) = x$.
