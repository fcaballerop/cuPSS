#ifndef HELPS_H
#define HELPS_H

// #ifdef WITHCUDA
#include <cuda_runtime.h>
// #else
// struct float2  // might have to change with cuda
// {
//     float x; float y;
// };
// #endif
//
#include <iostream>
#include <string>
#include <vector>

struct system_constants
{
    // Integration constants
    int sx, sy, sz;
    float dx, dy, dz;
    float dt;

    // Output constants
    int writeEveryNSteps;
};

struct pres
{
    float preFactor = 0.0f; // Numerical prefactor
    int q2n = 0;            // Powers of laplacian
    int iqx = 0;            // x-derivatives (contains i)
    int iqy = 0;            // y-derivatives (contains i)
    int invq = 0;           // 1/|q| powers
};

struct full_term
{
    std::vector<pres> prefactors;
    std::vector<std::string> fields;
};

#endif

#ifndef DEFS_H
#define DEFS_H

// Global definitions
#define PI 3.1415926535f

enum integrators { EULER, RK2, RK4 };

enum NoiseType {GaussianWhite};

#endif
