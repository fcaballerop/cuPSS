## 02_cahn_hilliard

This code uses cuPSS to integrate the Cahn-Hilliard equation in two dimensions.

Specifically, for a field $\phi(r,t)$, it solves the equation

$$ \partial_t \phi = D \nabla^2\left(a\phi + b\phi^3\right - k \nabla^2\phi\right) + \nabla\cdot\Lambda $$

The domain is square, with periodic boundary conditions, and the initial condition is a polar-symmetric droplet with a higher concentration than the background.

In this example, $\Lambda$ is a delta-correlated noise.
