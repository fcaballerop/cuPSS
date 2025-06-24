## 08_boundary_conditions

These examples show a standard way of enforcing Dirichlet and Neumann boundary conditions by extending the space domain and enforcing even or odd symmetries respectively. 

In the Dirichlet case, the example further shows how to enforce arbitary Dirichlet conditions by using auxiliary functions. It solves diffusion for a variable $u$ in one dimension, subject to the Dirichlet BCs $u(0)=A$ and $u(L)=B$. To do so, the solver expands space to $x\in(L,2L)$ and imposes odd symmetry around the boundary $x=L$. We introduce an auxiliary function $v$ and a correction function $h = A(1-x/L) + B x/L$, such that $v = u - h$.

Solving $\partial_t v = D\nabla^2v$ will enforce that $u$ follows a diffusion equation with the aforementioned Dirichlet BCs thanks to the correction function $h$.
