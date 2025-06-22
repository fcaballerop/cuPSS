## 06_kpz

This code integrates the KPZ equation. Unlike other examples, it doesn't output the state of the system at any point, but rather calculates the width for a given system size over a given number of runs. It is a good benchmark for the proper integratino of equation since it uses most cuPSS features and the KPZ exponents are known analytically.

This code will simulate, for a height field $h$, the equation

$$ \partial_t h = \nabla^2 h + \lambda (\nabla h)^2 + \eta $$

where the noise $\eta$ is delta-correlated. It will initialize the height at $h=0$ everywhere, and integrate the equation, calculating the width squared:

$$ W^2 = \frac{1}{L}\int dx (h(x) - \bar h)^2 $$

as a function of time. It will average this width over 'num_experiments' different noise realizations, and print the result on standard output.

The system size is fed a command line argument. If ran over several system sizes, one should observe that the initial growth evolves in time with a power law $W^2\sim t^{2/3}$, and the width will saturate to a value that depends on system size with a different critical exponent $W^2\sim L$.
