#ifndef TERM_KERNELS_HU
#define TERM_KERNELS_HU

#include <cuda_runtime.h>
#include "defines.h"

extern "C" void computeProduct_gpu(float2 **, float2 *out, int prodSize, int sx, int sy);

extern "C" void applyPrefactor_gpu(float2 *out, float pref, int q2n, int iqx, int iqy, int invq, int sx, int sy, float stepqx, float stepqy);

extern "C" void copyComp_gpu(float2 *, float2 *, int, int);

extern "C" void applyPres_vector_gpu(float2 *out, pres *prefactors, int p_len, int sx, int sy, float stepqx, float stepqy);

extern "C" void applyPres_vector_pre_gpu(float2 *out, float *pres, int mult_by_i, int sx, int sy);

__global__ void computeProduct_k(float2 **product, float2 *out, int prodSize, int sx, int sy);

__global__ void applyPrefactors_k(float2 *out, float pref, int q2n, int iqx, int iqy, int invq, int sx, int sy, float stepqx, float stepqy);

__global__ void copyComp_k(float2 *, float2 *, int, int);

__global__ void applyPres_vector_k(float2 *out, pres *prefactors, int p_len, int sx, int sy, float stepqx, float stepqy);

__global__ void applyPres_vector_pre_k(float2 *out, float *pres, int mult_by_i, int sx, int sy);


#endif
