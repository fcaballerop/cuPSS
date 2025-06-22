## 04_modelh

This code uses cuPSS to integrate the Model H equation in three dimensions.

Specifically, for a field $\phi(r,t)$, it solves the equation

$$ \partial_t \phi + v \cdot \nabla \phi= D \nabla^2\left(a\phi + b\phi^3\right - k \nabla^2\phi) + \nabla\cdot\Lambda$$

where $v$ is the velocity field of an incompressible Stokes' flow, that obeys an equation $\eta\nabla^2v - \nabla P +\nabla\cdot\sigma$, where $P$ is a pressure term set by incompressibility ($\nabla\cdot v = 0$), and $\nabla\cdot\sigma = -\mu\nabla\phi$ is a force given by capillarity forces.

There are two solvers in this example directory. One (*_base.cpp), solves the Stokes' flow equation by calculating the pressure.

The second (*_explicit_velocity.cpp) shows an advantage of Fourier methods in this case, since the flow can be solved formally in Fourier space, giving a local solution in frequency space: $v_i = \frac{1}{\eta q^2}\left(\delta_{ij} - q_iq_j/q^2\right)f_j$. This bypasses calculating the pressure and improves the speed.
