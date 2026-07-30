# 2D Integral Equations

This module extends the framework to solving higher-dimensional Integral Equations defined over a 2D spatial domain $(x, y) \in [0, 1] \times [0, 1]$.

## 2D Fredholm Integral Equations
- **Problem 11**: $u(x,y) = S(x,y) + \int_0^1 \int_0^1 (xs) u(s,t) ds dt$
  - Exact Solution: $u(x,y) = xy$

## 2D Volterra Integral Equations
- **Problem 12**: $u(x,y) = S(x,y) + \int_0^y \int_0^x u(s,t) ds dt$
  - Exact Solution: $u(x,y) = x+y$
