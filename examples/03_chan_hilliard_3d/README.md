## 03_cahn_hilliard_3d

This code uses cuPSS to integrate the Cahn-Hilliard equation in three dimensions.

Specifically, for a field $\phi(r,t)$, it solves the equation

$$ \partial_t \phi = D \nabla^2\left(a\phi + b\phi^3\right - k \nabla^2\phi) + \nabla\cdot\Lambda$$
