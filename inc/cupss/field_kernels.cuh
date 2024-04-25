#ifndef FIELD_KERNELS
#define FIELD_KERNELS

#include <cuda_runtime.h>
#include "defines.h"

extern "C" void setNotDynamic_gpu(float2 **, int , pres *, int , float2 *, int , int , float , float, float * , dim3, dim3);

extern "C" void setDynamic_gpu(float2 **, int , pres *, int , float2 *, int , int , float , float , float, float *, bool, float2 *, float *, dim3, dim3);

extern "C" void normalize_gpu(float2 *, int , int , dim3, dim3);

extern "C" void createNoise_gpu(float *, float *, int , int , float *, dim3, dim3);

extern "C" void dealias_gpu(float2 *, float2 *, int , int , int , dim3, dim3);

extern "C" void copyToFloat2_gpu(float *, float2 *, int , int , dim3, dim3);

extern "C" void correctNoiseAmplitude_gpu(float2 *, float *, int , int , dim3, dim3);

__global__ void correctNoiseAmplitude_k(float2 *, float *, int , int );

__global__ void normalize_k(float2 *, int , int );

__global__ void setNotDynamic_k(float2 **, int , pres *, int , float2 *, int , int , float , float, float *);

__global__ void setDynamic_k(float2 **, int , pres *, int , float2 *, int , int , float , float , float , float *, bool, float2 *, float *);

__global__ void createNoise_k(float *, float *, int , int , float *);

__global__ void copyToFloat2_k(float *, float2 *, int , int );

__global__ void conjNoise_k(float *, float *, int , int );

__global__ void dealias_k(float2 *, float2 *, int , int , int );

#endif
