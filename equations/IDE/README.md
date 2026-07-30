# Integro-Differential Equations (IDEs)

This module handles 1D equations containing both integrals and derivatives of the unknown function $u(x)$ simultaneously. These problems strictly require Boundary/Initial Conditions to converge to a unique solution.

## Fredholm IDEs
- **Problem 21**: $u''(x) = S(x) + \int_0^1 u(t) dt$
  - Exact Solution: $u(x) = e^x$
  - Boundary Conditions: $u(0) = 1$, $u(1) = e$

## Volterra IDEs
- **Problem 22**: $u'(x) = S(x) + \int_0^x u(t) dt$
  - Exact Solution: $u(x) = \sin(x)$
  - Initial Condition: $u(0) = 0$

- **Problem 24 (Non-linear)**: $u'(x) = S(x) + \int_0^x u^2(t) dt$
  - Exact Solution: $u(x) = x$
  - Initial Condition: $u(0)=0$

- **Problem 25 (Linear, 2nd Order)**: $u''(x) = S(x) + \int_0^x (x-t) u(t) dt$
  - Exact Solution: $u(x) = \sin(x)$
  - Boundary Conditions: $u(0) = 0, u(1) = \sin(1)$
