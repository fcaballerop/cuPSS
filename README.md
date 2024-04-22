<br />
<div align="center">
    <img src="img/noisy_gaussian.jpg" width=156 height=160>

  <h3 align="center">cuPPS</h3>

  <p align="center">
    A pseudospectral solver for systems of stochastics PDEs written in C++ & CUDA
    <br />
    <a href="https://github.com/fcaballerop/cuPSS/wiki"><strong>Read the docs »</strong></a>
    <br />
    <br />
  </p>
</div>

## Introduction

This library provides a framework for numerically integrating systems of 1st order PDEs (stochastic or deterministic) in a rectangular lattice, using a pseudospectral method. The integrator runs on NVIDIA GPUs through CUDA, but also supports running entirely on CPU, which can be faster for certain smaller lattices.

Detailed information about how to code a solver using this library can be found in the <a href="https://github.com/fcaballerop/cuPSS/wiki"><strong>docs</strong></a>.

## Dependencies

 * CUDA toolkit (11+)
 * cuFFT
 * cuRAND
 * FFTW3

## Quickstart

To run the cuPSS examples, first clone the repository
```
git clone https://github.com/fcaballerop/cuPSS.git
```
Then compile the library, (check dependencies above if any errors come up during compilation)
```
cd src/
nvcc -c *cpp -DWITHCUDA -O2
nvcc -c *cu -DWITHCUDA -O2
ar rcs libcupss.a *o
cd ../
```
This will create a file `src/libcupss.a` which can the be linked to any particular solver.

A number of example solvers can be found in the `examples` directory. They can be compiled by compiling with `nvcc`, linking the relevant libraries. For instance, the solver for model B, contained in `examples/modelb.cpp`, can be compiled by
```
nvcc examples/modelb.cpp -Lsrc/ -lcufft -lfftw3f -lcurand -lcupss -O2 -o modelb
```

The linking flag `-Lsrc/` should be changed to wherever `libcupss.a` is located. It will be in `src/` if the library has been compiled with the lines above. The solver can be run by
```
mkdir data
./modelb
```
The solver outputs data by default to a directory called `data` from where it's called. The output files contain raw data of the states of the field at each timestep at which they're written out.

<img src="img/modelb.gif" alt="Model B">

## What it calculates
See <a href=""><strong>here</strong></a>.

## What it supports
The solver supports any number of scalar fields $\lbrace\phi_i\rbrace$ on a rectangular lattice. Nonscalar fields such as vector fields must be split into as many scalar fields as components it has.

Each field $\phi_i$ can be updated through a dynamic rule, meaning it will solve an equation of the form
$$\partial_t\phi_i(t) = F_i[\lbrace\phi_j\rbrace]$$
or a static rule, meaning that each timestep the field is assigned a value as
$$\phi_i(t) = G_i[\lbrace\phi_j\rbrace]$$
Each field can be stochastic, so if it's updated through a dynamic rule, every timestep a noise term will be added according to:
$$\partial_t\phi_i(t) = F_i[\lbrace\phi_j\rbrace] + \eta_i$$
Where the noise is delta-correlated in space and time, or in Fourier space:
$$\langle\tilde\eta_i(k,\omega)\eta_j(k',\omega')\rangle = 2L(k)\delta(k+k')\delta(\omega+\omega')$$,
where $L(k)$ can be a constant or a power of wavelength $k$, allowing for several types of noise.

## What it does not support (yet)

 - Correlated noise.
 - 3D systems.
 - Compilable without CUDA so that it can run only on CPU.
 - Option for double precision integration.

<br />
<div align="center">
    <img src="img/diffusion.gif">
</div>
