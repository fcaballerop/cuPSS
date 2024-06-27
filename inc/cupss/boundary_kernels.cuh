#pragma once
#include <cuda_runtime.h>
#include "defines.h"

extern "C" void applyDiricheltSingleValue_gpu(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, dim3 blocks, dim3 threads_per_block);
extern "C" void applyDirichletMultipleValue_gpu(float2 * field_values,float * value,int depth,int dimension, bool leftwall, dim3 field_size , dim3 boundary_size,dim3 blocks, dim3 threads_per_block);
extern "C" void applyVonNuemannSingleValue_gpu(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h,dim3 blocks, dim3 threads_per_block);
extern "C" void applyVonNuemannMultipleValue_gpu(float2 * field_values,float * value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h,dim3 blocks, dim3 threads_per_block);

__global__ void applyDirichletSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size);
__global__ void applyDirichletMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall, dim3 field_size , dim3 boundary_size);
__global__ void applyVonNuemannSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h);
__global__ void applyVonNuemannMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h);
