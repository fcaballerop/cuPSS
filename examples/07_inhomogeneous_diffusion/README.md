## 07_inhomogeneous_diffusion

This example shows how to implement inhomogeneous operators. In this case, this code solves a diffusion equation with a diffusion constant that is a function of space, and with Neumann boundary conditions. 

The diffusion constant is chosen so that it's higher on the right half of the system $x>L/2$, following a hyperbolic tangent profile, i.e. $D(x) \sim D_0 + \tanh((x - L/2)/\xi)$, with a given variation distance $\xi$.

The Neumann boundary conditions are implemented by extending the system to $x\in(L, 2L)$, and enforcing even symmetry in the extended space, thus making sure that the derivatives of $\phi$ vanish at $x=0,L,2L$.
