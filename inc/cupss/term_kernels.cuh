#ifndef TERM_KERNELS_HU
#define TERM_KERNELS_HU

#include <cuda_runtime.h>
#include "defines.h"

extern "C" void computeProduct_gpu(float2 **, float2 *, int , int , int , int, dim3, dim3);

extern "C" void applyPrefactor_gpu(float2 *, float , int , int , int , int, int , int , int , int, float, float , float , dim3, dim3);

extern "C" void copyComp_gpu(float2 *, float2 *, int, int, int, dim3, dim3);

extern "C" void applyPres_vector_gpu(float2 *, pres *, int , int , int , int, float, float , float , dim3, dim3);

extern "C" void applyPres_vector_pre_gpu(float2 *, float *, int , int , int , int, dim3, dim3);

__global__ void computeProduct_k(float2 **, float2 *, int , int , int , int);

__global__ void applyPrefactors_k(float2 *, float , int , int , int, int, int , int , int , int , float , float, float );

__global__ void copyComp_k(float2 *, float2 *, int, int, int);

__global__ void applyPres_vector_k(float2 *, pres *, int , int , int , int, float, float , float );

__global__ void applyPres_vector_pre_k(float2 *, float *, int , int , int , int);


#endif
