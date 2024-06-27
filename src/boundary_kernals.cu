#include <cuda_runtime.h>
#include <iostream>

#include "../inc/cupss/boundary_kernals.cuh"


__global__ void applyDirichletSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,int sx, int sy, int sz,int iter_x, int iter_y, int iter_z) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = sx;
    size[1] = sy;
    size[2] = sz;
    if (dim_i[0] < sx && dim_i[1] < sy && dim_i[2] < sz)
    {
        int index;
        for (int ib = 0; ib<depth; ib++){
            if (leftwall){
                dim_i[dimension] = ib;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
            }
            index = dim_i[0] + dim_i[1] * sx + dim_i[2] * sx * sy;
            field_values[index].x = value;
        }
    }
}
__global__ void applyDirichletMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,int sx, int sy, int sz, int iter_x, int iter_y, int iter_z) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = sx;
    size[1] = sy;
    size[2] = sz;
    if (dim_i[0] < iter_x && dim_i[1] < iter_y && dim_i[2] < iter_z) {
        int valueIndex = dim_i[0] + dim_i[1]*iter_x + dim[2]*iter_x*iter_y;
        int index;
        for (int ib = 0; ib<depth; ib++){
            if (leftwall){
                dim_i[dimension] = ib;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
            }
            index = dim_i[0] + dim_i[1] * sx + dim_i[2] * sx * sy;
            field_values[index].x = value[valueIndex];
        }
    }
}
__global__ void applyVonNuemannSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,int sx, int sy, int sz, int iter_x, int iter_y, int iter_z, float h) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = sx;
    size[1] = sy;
    size[2] = sz;
    int dim_i_one_in[3]
    if (dim_i[0] < sx && dim_i[1] < sy && dim_i[2] < sz)
    {
        int index;
        int index_one_in;
        for (int ib = depth-1; ib>=0; ib--){
            if (leftwall){
                dim_i[dimension] = ib;
                dim_i_one_in[dimension] = ib+1;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
                dim_i_one_in[dimension] = size[dimension] - 1 - ib -1;

            }
            index = dim_i[0] + dim_i[1] * sx + dim_i[2] * sx * sy;
            index_one_in = dim_i_one_in[0] + dim_i_one_in[1] * sx + dim_i_one_in[2] * sx * sy;
            field_values[index].x =  field_values[index_one_in].x - h * value;
        }
    }
}
__global__ void applyVonNuemannMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,int sx, int sy, int sz, int iter_x, int iter_y, int iter_z, float h) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = sx;
    size[1] = sy;
    size[2] = sz;
    if (dim_i[0] < iter_x && dim_i[1] < iter_y && dim_i[2] < iter_z) {
        int valueIndex = dim_i[0] + dim_i[1]*iter_x + dim[2]*iter_x*iter_y;
        int index;
        for (int ib = depth-1; ib>=0; ib--){
            if (leftwall){
                dim_i[dimension] = ib;
                dim_i_one_in[dimension] = ib+1;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
                dim_i_one_in[dimension] = size[dimension] - 1 - ib -1;
            }
            index = dim_i[0] + dim_i[1] * sx + dim_i[2] * sx * sy;
            index_one_in = dim_i_one_in[0] + dim_i_one_in[1] * sx + dim_i_one_in[2] * sx * sy;

            field_values[index].x =  field_values[index_one_in].x - h *value[valueIndex];
        }
    }
}
