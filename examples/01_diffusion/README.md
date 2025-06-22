## 01_diffusion

This code uses cuPSS to integrate a basic homogeneous diffusion in two dimensions.

Specifically, for a field phi, it solves the equation

$$ \partial_t \phi(r, t) = D \nabla^2\phi(r,t) $$

The domain is square, with periodic boundary conditions, and the initial condition is a polar-symmetric droplet with a higher concentration than the background.
