<br />
<div align="center">
    <img src="img/noisy_gaussian.jpg" width=156 height=160>

  <h3 align="center">cuPSS</h3>

  <p align="center">
    A pseudospectral solver for systems of stochastics PDEs written in C++ & CUDA
      <br/>
    <br />
    <a href="https://github.com/fcaballerop/cuPSS/wiki/Tutorials"><strong>Read the tutorials »</strong></a>
      <br />
      <br />
    <a href="https://arxiv.org/abs/2405.02410"><strong>Read the preprint »</strong></a>
    <br />
    <br />
  </p>
</div>

## Introduction

This library provides a framework for numerically integrating systems of PDEs (stochastic or deterministic) pseudo-spectrally. The integrator runs on NVIDIA GPUs through CUDA, but also supports running entirely on CPU, which can be faster for certain smaller systems.

Detailed information about how to code a solver using this library can be found in the <a href="https://github.com/fcaballerop/cuPSS/wiki/Tutorials"><strong>tutorials</strong></a>.

cuPSS has been tested on Linux, Windows and Windows WSL. Note that Windows WSL does not support CUDA/OpenGL interoperability and thus real time visualization on WSL is not possible.

## Dependencies

 * CUDA toolkit (11+)
 * FFTW3 (with single point precision)

## Quickstart

### Installing dependencies

Install the <a href="https://developer.nvidia.com/cuda-toolkit">CUDA toolkit</a>. 

Install <a href="https://www.fftw.org/">fftw3</a> with single point precision. Follow their <a href="https://www.fftw.org/fftw3_doc/Installation-on-Unix.html">installation instructions</a> adding `--enable-float` to the `./configure` command.

### Compiling the library

To download cuPSS and test the examples, clone the repository
```
git clone https://github.com/fcaballerop/cuPSS.git
cd cuPSS/
```

#### Option 1: Compilation with global installation
A CMake file is given for convenience, so cuPSS can be installed system wide by running (check dependencies above if any errors come up during compilation, specially fftw3 with single point precision).
```
mkdir build
cd build
cmake ../
cmake --build .
sudo cmake --install .
```
This will create a file `build/libcupss.a` which can the be linked to any particular solver. The last command will copy this library to `/usr/lib`, and the header files to `/usr/include`. 

#### Option 2: Compilation with local installation
If CMake is not available, or system wide installation is not possible, the library can be compiled in place, by running from the root directory
```
cd src
nvcc -c *cu *cpp -O2
ar rcs libcupss.a *o
cd ..
```
This will create a file `src/libcupss.a` which can be linked to any particular solver. The header files are in `inc`.

### Compiling and running an example
A number of example solvers can be found in the `examples` directory. They can be compiled with `nvcc`, linking the relevant libraries. For instance, the solver for model B, contained in `examples/cahn-hilliars.cpp`, can be compiled by
```
nvcc examples/cahn-hilliard.cpp -lcufft -lfftw3f -lcurand -lcupss -O2 -o cahn-hilliard
```
If cupss was not installed globally, the location of `libcupss.a` must be specified with the linker flag `-L`:
```
nvcc examples/cahn-hilliard.cpp -Lsrc/ -lcufft -lfftw3f -lcurand -lcupss -O2 -o cahn-hilliard
```
The linking flag `-Lsrc/` should be changed to wherever `libcupss.a` is located. It will be in `src/` if the library has been compiled with the lines in *Option 2* above. The solver can be run with
```
./cahn-hilliard
```
The solver outputs data by default to a directory called `data` from where it's called. The output files contain raw data of the states of the field at each timestep at which they're written out, which can be plotted/analysed separately.

These two images are the results of the spinodal decomposition of the Cahn-Hilliard solver in 2D and 3D, available in the examples.
<div align="center">
    <img src="img/CH2D.gif" width=220 height=200>
    <img src="img/CH3D.gif" width=220 height=200>
</div>

### Run tests

There is a tests file that will run a set of unit tests for all differential operators and initialization in CPU and GPU. IT depends on the Google Testing suit, and can be run by running:
```
cd tests
nvcc tests.cpp -o tests -lcupss -lcufft -lcurand -lfftw3f -lgtest
./tests
```

## Troubleshooting
The CUDA toolkit does not add its binaries location to `PATH` by default. They're installed by default in `/usr/local/cuda/bin/`. Add that directory to `PATH` or substitute all calls to `nvcc` with `/usr/local/cuda/bin/nvcc`.

There are two main reasons why compilation of either the cuPSS library or a particular solver might fail.

 - Make sure fftw3 is available. This means compiling fftw3 from sources with single point precision, and then linking this version of the library `-lfftw3f` (notice the `f` at the end, for `float`). Single precision is not compiled automatically so needs to be specified. It should be enough to run `./configure --enable-float && make && sudo make install` from the directory of fftw3 sources.
 - Make sure the CUDA driver version and CUDA toolkit are compatible. The CUDA driver version should be equal or greater to your installed CUDA toolkit. The version of the CUDA driver can be checked by running `nvidia-smi`, which will report the `CUDA Version` at the top right. The CUDA toolkit version can be found by running `nvcc --version`, or `/usr/local/cuda/bin/nvcc --version` if installed in the default location and not added to `PATH`. Having a more recent CUDA toolkit version might compile just fine but produce an invalid library. Solvers will show a warning in this case.

## What it calculates
See <a href="https://github.com/fcaballerop/cuPSS/wiki#what-can-it-solve"><strong>here</strong></a>.

## What it does not support (yet)
 - Compilable without CUDA so that it can run only on CPU.
 - Option for double precision integration.

<br />
<div align="center">
    <img src="img/diffusion.gif">
</div>
