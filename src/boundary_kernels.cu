#include <cuda_runtime.h>
#include <iostream>

#include "boundary_kernels.cuh"
extern "C" void applyDiricheltSingleValue_gpu(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, dim3 blocks, dim3 threads_per_block){
    applyDirichletSingleValue<<<blocks, threads_per_block>>>(field_values,value,depth,dimension,leftwall,field_size,boundary_size);
}

extern "C" void applyDiricheltMultipleValue_gpu(float2 * field_values,float * value,int depth,int dimension, bool leftwall, dim3 field_size , dim3 boundary_size,dim3 blocks, dim3 threads_per_block){
    applyDirichletMultipleValue<<<blocks, threads_per_block>>>(field_values,value,depth,dimension,leftwall,field_size,boundary_size);

}
extern "C" void applyVonNuemannSingleValue_gpu(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h,dim3 blocks, dim3 threads_per_block){
    applyVonNuemannSingleValue<<<blocks, threads_per_block>>>(field_values,value,depth,dimension,leftwall,field_size,boundary_size,h);

}
extern "C" void applyVonNuemannMultipleValue_gpu(float2 * field_values,float * value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h,dim3 blocks, dim3 threads_per_block){
    applyVonNuemannMultipleValue<<<blocks, threads_per_block>>>(field_values,value,depth,dimension,leftwall,field_size,boundary_size,h);

}


__global__ void applyDirichletSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = field_size.x;
    size[1] = field_size.y;
    size[2] = field_size.z;
    if (dim_i[0] < boundary_size.x && dim_i[1] < boundary_size.y && dim_i[2] < boundary_size.z) {
        int index;
        for (int ib = 0; ib<depth; ib++){
            if (leftwall){
                dim_i[dimension] = ib;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
            }
            index = dim_i[0] + dim_i[1] * field_size.x + dim_i[2] * field_size.x * field_size.y;
            field_values[index].x = value;
        }
    }
}
__global__ void applyDirichletMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = field_size.x;
    size[1] = field_size.y;
    size[2] = field_size.z;
    if (dim_i[0] < boundary_size.x && dim_i[1] < boundary_size.y && dim_i[2] < boundary_size.z) {
        int valueIndex = dim_i[0] + dim_i[1]*boundary_size.x + dim_i[2]*boundary_size.x*boundary_size.y;
        int index;
        for (int ib = 0; ib<depth; ib++){
            if (leftwall){
                dim_i[dimension] = ib;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
            }
            index = dim_i[0] + dim_i[1] * field_size.x + dim_i[2] * field_size.x * field_size.y;
            field_values[index].x = value[valueIndex];
        }
    }
}
__global__ void applyVonNuemannSingleValue(float2 * field_values,float value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h) {
    int dim_i[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = field_size.x;
    size[1] = field_size.y;
    size[2] = field_size.z;
    int dim_i_one_in[3];
    if (dim_i[0] < boundary_size.x && dim_i[1] < boundary_size.y && dim_i[2] < boundary_size.z) {
        int fieldIndex;
        int index_one_in;
        dim_i_one_in[0]=dim_i[0];
        dim_i_one_in[1]=dim_i[1];
        dim_i_one_in[2]=dim_i[2];
        for (int ib = depth-1; ib>=0; ib--){
            if (leftwall){
                dim_i[dimension] = ib;
                dim_i_one_in[dimension] = ib+1;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib; //5 -1 - 1 = 3
                dim_i_one_in[dimension] = size[dimension] - 1 - (ib +1); // 5 -1 -1 -1 = 2

            }
            fieldIndex = dim_i[0] + dim_i[1] * field_size.x + dim_i[2] * field_size.x * field_size.y;
            index_one_in = dim_i_one_in[0] + dim_i_one_in[1] * field_size.x + dim_i_one_in[2] * field_size.x * field_size.y;
            field_values[fieldIndex].x =  field_values[index_one_in].x - h * value;
        }
    }
}
__global__ void applyVonNuemannMultipleValue(float2 * field_values,float * value,int depth,int dimension, bool leftwall,dim3 field_size, dim3 boundary_size, float h) {
    int dim_i[3];
    int dim_i_one_in[3];
    int size[3];
    dim_i[0] = blockIdx.x * blockDim.x + threadIdx.x;
    dim_i[1] = blockIdx.y * blockDim.y + threadIdx.y;
    dim_i[2] = blockIdx.z * blockDim.z + threadIdx.z;
    size[0] = field_size.x;
    size[1] = field_size.y;
    size[2] = field_size.z;
    if (dim_i[0] < boundary_size.x && dim_i[1] < boundary_size.y && dim_i[2] < boundary_size.z) {
        int valueIndex = dim_i[0] + dim_i[1]*boundary_size.x + dim_i[2]*boundary_size.x*boundary_size.y;
        int fieldIndex;
        int index_one_in;
        dim_i_one_in[0]=dim_i[0];
        dim_i_one_in[1]=dim_i[1];
        dim_i_one_in[2]=dim_i[2];
        for (int ib = depth-1; ib>=0; ib--){
            if (leftwall){
                dim_i[dimension] = ib;
                dim_i_one_in[dimension] = ib+1;
            } else {
                dim_i[dimension] = size[dimension] - 1 - ib;
                dim_i_one_in[dimension] = size[dimension] - 1 - ib -1;
            }
            fieldIndex = dim_i[0] + dim_i[1] * field_size.x + dim_i[2] * field_size.x * field_size.y;
            index_one_in = dim_i_one_in[0] + dim_i_one_in[1] * field_size.x + dim_i_one_in[2] * field_size.x * field_size.y;

            field_values[fieldIndex].x =  field_values[index_one_in].x - h *value[valueIndex];
        }
    }
}
