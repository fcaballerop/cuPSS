#include <cuda_runtime.h>
#include <iostream>

#include "../inc/cupss/field_kernels.cuh"

extern "C" void setNotDynamic_gpu(float2 **terms, int len, pres *implicits, int i_len, float2 *out, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz, float *precomp, bool isNoisy, float2 *noise, float *precomp_noise, float dt, dim3 blocks, dim3 threadsPerBlock)
{
    setNotDynamic_k<<<blocks, threadsPerBlock>>>(terms, len, implicits, i_len, out, sx, sy, sz, stepqx, stepqy, stepqz, precomp, isNoisy, noise, precomp_noise, dt);
}

extern "C" void setDynamic_gpu(float2 **terms, int len, pres *implicits, int i_len, float2 *out, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz, float dt, float *precomp, bool isNoisy, float2 *noise_r, float* precomp_noise, dim3 blocks, dim3 threadsPerBlock)
{
    setDynamic_k<<<blocks, threadsPerBlock>>>(terms, len, implicits, i_len, out, sx, sy, sz, stepqx, stepqy, stepqz, dt, precomp, isNoisy, noise_r, precomp_noise);
}

extern "C" void createNoise_gpu(float *noise_comp_d_r, float *noise_comp_d_i, int sx, int sy, int sz, float *amplitude, dim3 blocks, dim3 threadsPerBlock)
{
    cudaDeviceSynchronize();
    createNoise_k<<<blocks,threadsPerBlock>>>(noise_comp_d_r, noise_comp_d_i, sx, sy, sz, amplitude);
    cudaDeviceSynchronize();
    conjNoise_k<<<blocks,threadsPerBlock>>>(noise_comp_d_r, noise_comp_d_i, sx, sy, sz);
}

extern "C" void dealias_gpu(float2 *comp_array_d, float2 *comp_dealiased_d, int sx, int sy, int sz, int aliasing_order, dim3 blocks, dim3 threadsPerBlock)
{
    dealias_k<<<blocks, threadsPerBlock>>>(comp_array_d, comp_dealiased_d, sx, sy, sz, aliasing_order);
}

extern "C" void copyToFloat2_gpu(float *in, float2 *out, int sx, int sy, int sz, dim3 blocks, dim3 threadsPerBlock)
{
    copyToFloat2_k<<<blocks, threadsPerBlock>>>(in, out, sx, sy, sz);
}

extern "C" void correctNoiseAmplitude_gpu(float2 *noise_fourier, float *precomp_noise_d, int sx, int sy, int sz, dim3 blocks, dim3 threadsPerBlock)
{
    correctNoiseAmplitude_k<<<blocks, threadsPerBlock>>>(noise_fourier, precomp_noise_d, sx, sy, sz);
}

extern "C" void normalize_gpu(float2 *array, int sx, int sy, int sz, dim3 blocks, dim3 threadsPerBlock)
{
    normalize_k<<<blocks, threadsPerBlock>>>(array, sx, sy, sz);
}

__global__ void correctNoiseAmplitude_k(float2 *noise_fourier, float *precomp_noise_d, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        noise_fourier[index].x *= precomp_noise_d[index];
        noise_fourier[index].y *= precomp_noise_d[index];
    }
}

__global__ void copyToFloat2_k(float *in, float2 *out, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        out[index].x = in[index];
        out[index].y = 0.0f;
    }
}

__global__ void createNoise_k(float *noise_comp_d_r, float *noise_comp_d_i, int sx, int sy, int sz, float *amplitude)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        noise_comp_d_r[index] *= amplitude[index];
        noise_comp_d_i[index] *= amplitude[index];
    }
}

__global__ void conjNoise_k(float *noise_comp_d_r, float *noise_comp_d_i, int sx, int sy, int sz)
{
    // THIS DOES NOT WORK IN 3D
    // IT'S ALSO NOT IN USE
    // NOISE IS CREATED IN REAL SPACE AND FOURIER TRANSFORMED
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        if (i >= sx/2 || j >= sy/2)
        {
            int n_i = sx - i;
            int n_j = sy > 1 ? sy - j : 0;
            int n_k = sz > 1 ? sz - k : 0;
            int n_index = n_k * sx * sy + n_j * sx + n_i;
            noise_comp_d_r[index] = noise_comp_d_r[n_index];
            noise_comp_d_i[index] = - noise_comp_d_i[n_index];
        }
        // hacky for now, only works in 1d
        if (i == 0 || (sx%2==0 && i == sx/2))
            noise_comp_d_i[index] = 0.0f;
        // if ((i == 0 || (sx%2 == 0 && i == sx/2)) && (sy > 0 && (j == 0 || (sx%2 == 0 && j == sy/2))))
        //     noise_comp_d_i[index] = 0.0f;
    }
}

__global__ void normalize_k(float2 *array, int sx, int sy, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        float normalization = 1.0f / ((float)(sx*sy*sz));
        array[index].x *= normalization;
        array[index].y = 0.0f;
    }
}

__global__ void setNotDynamic_k(float2 **terms, int len, pres *implicits, int i_len, float2 *out, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz, float *precomp, bool isNoisy, float2 *noise, float *precomp_noise, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        int t = 0;
        for (t = 0; t < len; t++)
        {
            if (t == 0)
            {
                out[index].x = terms[t][index].x;
                out[index].y = terms[t][index].y;
            }
            else
            {
                out[index].x += terms[t][index].x;
                out[index].y += terms[t][index].y;
            }
        }
        if (isNoisy)
        {
            float sdt = 1.0f / sqrt(dt);
            if (len == 0)
            {
                out[index].x = sdt * precomp_noise[index] * noise[index].x;
                out[index].y = sdt * precomp_noise[index] * noise[index].y;
            }
            else 
            {
                out[index].x += sdt * precomp_noise[index] * noise[index].x;
                out[index].y += sdt * precomp_noise[index] * noise[index].y;
            }
        }
        if (i_len > 0 && index > 0)
        {
            float implicitFactor = 0.0f;
            float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
            float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
            float qz = (k < (sz+1)/2 ? (float)k : (float)(k - sz)) * stepqz;
            float q2 = qx*qx + qy*qy + qz*qz;

            for (t = 0; t < i_len; t++)
            {
                float thisImplicit = implicits[t].preFactor;
                if (implicits[t].q2n != 0)
                    thisImplicit *= pow(q2, implicits[t].q2n);
                if (implicits[t].invq != 0)
                {
                    float invq = 0.0f;
                    if (index > 0)
                        invq = 1.0f / sqrt(q2);
                    thisImplicit *= pow(invq, implicits[t].invq);
                }
                implicitFactor += thisImplicit;
            }
            if (index == 0 && abs(implicitFactor) < 0.000001f) // meant to be epsilon
                implicitFactor = 1.0f;  // Do nothing
            out[index].x /= implicitFactor;
            out[index].y /= implicitFactor;
            // out[index].x /= precomp[index];
            // out[index].y /= precomp[index];
        }
    }
}

__global__ void setDynamic_k(float2 **terms, int len, pres *implicits, int i_len, float2 *out, int sx, int sy, int sz, float stepqx, float stepqy, float stepqz, float dt, float *precomp, bool isNoisy, float2 *noise, float *precomp_noise)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        int t = 0;
        for (t = 0; t < len; t++)
        {
            out[index].x += dt * terms[t][index].x;
            out[index].y += dt * terms[t][index].y;
        }
        if (isNoisy)
        {
            // includes sqrt(dt) in the precomp_noise
            out[index].x += precomp_noise[index] * noise[index].x;
            out[index].y += precomp_noise[index] * noise[index].y;
        }
        // Now implicit terms
        if (i_len > 0)
        {
            out[index].x /= precomp[index];
            out[index].y /= precomp[index];
        }
    }
}

__global__ void dealias_k(float2 *comp_array_d, float2 *comp_dealiased_d, int sx, int sy, int sz, int aliasing_order)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int index = k * sx * sy + j * sx + i;
    if (i < sx && j < sy && k < sz)
    {
        int n_i = i;
        int n_j = j;
        int n_k = k;
        if (i > sx/2) n_i -= sx;
        if (j > sy/2) n_j -= sy;
        if (k > sz/2) n_k -= sz;

        if (abs(n_i) > sx/(aliasing_order+1) || abs(n_j) > sy/(aliasing_order+1) || abs(n_k) > sz/(aliasing_order+1))
        {
            comp_dealiased_d[index].x = 0.0f;
            comp_dealiased_d[index].y = 0.0f;
        }
        else
        {
            comp_dealiased_d[index].x = comp_array_d[index].x;
            comp_dealiased_d[index].y = comp_array_d[index].y;
        }
    }
}
