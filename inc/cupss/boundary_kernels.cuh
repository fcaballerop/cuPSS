#pragma once
#include <cuda_runtime.h>
#include "defines.h"

extern "C" void



__global__ void applyDirichletSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,int sx, int sy, int sz,int iter_x, int iter_y, int iter_z);
__global__ void applyDirichletMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,int sx, int sy, int sz, int iter_x, int iter_y, int iter_z);
__global__ void applyVonNuemannSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,int sx, int sy, int sz, int iter_x, int iter_y, int iter_z, float h);
__global__ void applyVonNuemannMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,int sx, int sy, int sz, int iter_x, int iter_y, int iter_z, float h);
