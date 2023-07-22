#ifndef FIELD_KERNELS
#define FIELD_KERNELS

#include <cuda_runtime.h>
#include "defines.h"

extern "C" void setNotDynamic_gpu(float2 **, int , pres *, int , float2 *, int , int , float , float, float * );

extern "C" void setDynamic_gpu(float2 **, int , pres *, int , float2 *, int , int , float , float , float, float *, bool, float2 *, float *);

extern "C" void normalize_gpu(float2 *, int , int );

extern "C" void createNoise_gpu(float *, float *, int , int , float *);

extern "C" void dealias_gpu(float2 *comp_array_d, float2 *comp_dealiased_d, int sx, int sy, int aliasing_order);

extern "C" void copyToFloat2_gpu(float *in, float2 *out, int sx, int sy);

extern "C" void correctNoiseAmplitude_gpu(float2 *noise_fourier, float *precomp_noise_d, int sx, int sy);

__global__ void correctNoiseAmplitude_k(float2 *noise_fourier, float *precomp_noise_d, int sx, int sy);

__global__ void normalize_k(float2 *array, int sx, int sy);

__global__ void setNotDynamic_k(float2 **, int , pres *, int , float2 *, int , int , float , float, float *);

__global__ void setDynamic_k(float2 **terms, int len, pres *implicits, int i_len, float2 *out, int sx, int sy, float stepqx, float stepqy, float dt, float *, bool, float2 *, float *);

__global__ void createNoise_k(float *, float *, int , int , float *);

__global__ void copyToFloat2_k(float *in, float2 *out, int sx, int sy);

__global__ void conjNoise_k(float *, float *, int , int );

__global__ void dealias_k(float2 *comp_array_d, float2 *comp_dealiased_d, int sx, int sy, int aliasing_order);

#endif
