#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#include "../inc/cupss/term_kernels.cuh"

extern "C" void computeProduct_gpu(float2 **product, float2 *out, int prodSize, int sx, int sy, int sz, dim3 blocks, dim3 threadsPerBlock)
{
    computeProduct_k<<<blocks, threadsPerBlock>>>(product, out, prodSize, sx, sy, sz);
}

extern "C" void applyPrefactor_gpu(float2 *out, float pref, int q2n, int iqx, int iqy, int iqz, int invq, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz, dim3 blocks, dim3 threadsPerBlock)
{
    applyPrefactors_k<<<blocks, threadsPerBlock>>>(out, pref, q2n, iqx, iqy, iqz, invq, sx, sy, sz, stepqx, stepqy, stepqz);
}

extern "C" void copyComp_gpu(float2 *out, float2 *in, int sx, int sy, int sz, dim3 blocks, dim3 threadsPerBlock)
{
    copyComp_k<<<blocks, threadsPerBlock>>>(out, in, sx, sy, sz);
}

extern "C" void applyPres_vector_gpu(float2 *out, pres *prefactors, int p_len, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz, dim3 blocks, dim3 threadsPerBlock)
{
    applyPres_vector_k<<<blocks, threadsPerBlock>>>(out, prefactors, p_len, sx, sy, sz, stepqx, stepqy, stepqz);
}

extern "C" void applyPres_vector_pre_gpu(float2 *out, float *pres, int mult_by_i, int sx, int sy, int sz, dim3 blocks, dim3 threadsPerBlock)
{
    applyPres_vector_pre_k<<<blocks, threadsPerBlock>>>(out, pres, mult_by_i, sx, sy, sz);
}

__global__ void copyComp_k(float2 *out, float2 *in, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        out[index].x = in[index].x;
        out[index].y = in[index].y;
    }
}

__global__ void computeProduct_k(float2 **product, float2 *out, int prodSize, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        int c;
        float result = 1.0f;
        for (c = 0; c < prodSize; c++)
        {
            result *= product[c][index].x;
        }
        out[index].x = result;
        out[index].y = 0.0f;
        // if (i == 0 && j == 0 && k ==0)
        // {
        //     printf("Prod size %d, resul %f, origin %f\n", prodSize, result, product[0][index].x);
        // }
    }
}

__global__ void applyPrefactors_k(float2 *out, float pref, int q2n, int iqx, int iqy, int iqz, int invq, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;

    if (i < sx && j < sy && k < sz)
    {
        out[index].x *= pref;
        out[index].y *= pref;

        float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
        float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
        float qz = (k < (sz+1)/2 ? (float)k : (float)(k - sz)) * stepqz;
        float q2 = qx*qx + qy*qy + qz*qz;

        if (q2n > 0)
        {
            float powerq2n = pow(q2, q2n);
            out[index].x *= powerq2n;
            out[index].y *= powerq2n;
        }
        if (iqx > 0)
        {
            int n2 = iqx % 2;
            int a = (iqx - n2) / 2;
            int m1power = 2 * (- a % 2) + 1;
            float qpower = qx;
            if (iqx > 1) qpower = pow(qx, iqx);
            out[index].x *= (float)m1power * qpower;
            out[index].y *= (float)m1power * qpower;
            if (n2 == 1)
            {
                float aux = out[index].x;
                out[index].x = - out[index].y;
                out[index].y = aux;
            }
        }
        if (iqy > 0)
        {
            int n2 = iqy % 2;
            int a = (iqy - n2) / 2;
            int m1power = 2 * (- a % 2) + 1;
            float qpower = qy;
            if (iqy > 1) qpower = pow(qy, iqy);
            out[index].x *= (float)m1power * qpower;
            out[index].y *= (float)m1power * qpower;
            if (n2 == 1)
            {
                float aux = out[index].x;
                out[index].x = - out[index].y;
                out[index].y = aux;
            }
        }
        if (iqz > 0)
        {
            int n2 = iqz % 2;
            int a = (iqz - n2) / 2;
            int m1power = 2 * (- a % 2) + 1;
            float qpower = qz;
            if (iqz > 1) qpower = pow(qy, iqz);
            out[index].x *= (float)m1power * qpower;
            out[index].y *= (float)m1power * qpower;
            if (n2 == 1)
            {
                float aux = out[index].x;
                out[index].x = - out[index].y;
                out[index].y = aux;
            }
        }
        if (invq > 0)
        {
            float invq_f = 0.0f;
            if (index > 0)
                invq_f = 1.0f / pow(sqrt(q2), invq);
            out[index].x *= invq_f;
            out[index].y *= invq_f;
        }
    }
}

__global__ void applyPres_vector_k(float2 *out, pres *prefactors, int p_len, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;

    if (i < sx && j < sy && k < sz)
    {
        float totalPrefactor = 0.0f;
        int multiply_by_i = (prefactors[0].iqx + prefactors[0].iqy + prefactors[0].iqz)%2;
        int p = 0;
        for (p = 0; p < p_len; p++)
        {
            float num_prefactor = prefactors[p].preFactor;
            int imaginary_units = prefactors[p].iqx + prefactors[p].iqy + prefactors[p].iqz;
            int multiply_by_i_this = imaginary_units % 2;
            if (multiply_by_i != multiply_by_i_this)
            {
                printf("PANIC: Inconsistent powers of I in prefactors (GPU)\n");
            }
            int negate = -2 * (((imaginary_units - multiply_by_i)/2)%2) + 1;
            num_prefactor *= (float)negate;
            
            float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
            float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
            float qz = (k < (sz+1)/2 ? (float)k : (float)(k - sz)) * stepqz;
            float q2 = qx*qx + qy*qy + qz*qz;

            if (prefactors[p].q2n > 0)
            {
                num_prefactor *= pow(q2, prefactors[p].q2n);
            }
            if (prefactors[p].iqx > 0)
            {
                num_prefactor *= pow(qx, prefactors[p].iqx);    
            }
            if (prefactors[p].iqy > 0)
            {
                num_prefactor *= pow(qy, prefactors[p].iqy);    
            }
            if (prefactors[p].iqz > 0)
            {
                num_prefactor *= pow(qz, prefactors[p].iqz);    
            }
            if (prefactors[p].invq > 0)
            {
                float invq_i = 0.0f;
                if (index > 0)
                    invq_i = 1.0f / sqrt(q2);
                num_prefactor *= pow(invq_i, prefactors[p].invq);    
            }
            totalPrefactor += num_prefactor;
        }
        out[index].x *= totalPrefactor;
        out[index].y *= totalPrefactor;
        if (multiply_by_i)
        {
            float aux = out[index].x;
            out[index].x = - out[index].y;
            out[index].y = aux;
        }
    }
}

__global__ void applyPres_vector_pre_k(float2 *out, float *pres, int mult_by_i, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;

    if (i < sx && j < sy && k < sz)
    {
        out[index].x *= pres[index];
        out[index].y *= pres[index];
        if (mult_by_i)
        {
            float aux = out[index].x;
            out[index].x = - out[index].y;
            out[index].y = aux;
        }
    }
}
