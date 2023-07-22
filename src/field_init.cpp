#include <cmath>
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <fftw3.h>
#include <cufft.h>
#include <driver_types.h>
#include <iostream>
#include <random>
#include <ostream>
#include "../inc/field.h"
#include "../inc/field_kernels.cuh"
#include "../inc/term.h"
#include "../inc/defines.h"

field::field(int _sx, int _sy, float _dx, float _dy) : sx(_sx), sy(_sy), dx(_dx), dy(_dy), stepqx(2.0f*PI/(_dx * (float)_sx)), stepqy(2.0f*PI/(_dy * (float)_sy)), rng(rd()), dist(std::normal_distribution<>(0, 1.0))
{
    rng.seed(time(NULL));
    // sx = _sx;
    // sy = _sy;
    // dx = _dx;
    // dy = _dy;
    // stepqx = 2.0f * PI / (dx * sx);
    // stepqy = 2.0f * PI / (dy * sy);
    real_array = new float2[sx*sy];
    comp_array = new float2[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&real_array_d), sx * sy * sizeof(float2));
    cudaMalloc(reinterpret_cast<void **>(&comp_array_d), sx * sy * sizeof(float2));

    precomp_implicit = new float[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&precomp_implicit_d), sx * sy * sizeof(float));

    precomp_noise = new float[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&precomp_noise_d), sx * sy * sizeof(float));

    noise_comp = new float2[sx*sy];
    noise_gend = new float2[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&noise_comp_d_r), sx * sy * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&noise_comp_d_i), sx * sy * sizeof(float));

    cudaMalloc(reinterpret_cast<void **>(&gen_noise), sx * sy * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&noise_real), sx * sy * sizeof(float2));
    cudaMalloc(reinterpret_cast<void **>(&noise_fourier), sx * sy * sizeof(float2));

    needsaliasing = false;
    aliasing_order = 1;
    comp_dealiased = new float2[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&comp_dealiased_d), sx * sy * sizeof(float2));
    real_dealiased = new float2[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&real_dealiased_d), sx * sy * sizeof(float2));

    outputToFile = true;

    // Initialize fftw plans
    if (sy == 1) // 1d plans
    {
        plan_forward = fftwf_plan_dft_1d(sx, 
                reinterpret_cast<fftwf_complex*>(real_array), 
                reinterpret_cast<fftwf_complex*>(comp_array), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftwf_plan_dft_1d(sx, 
                reinterpret_cast<fftwf_complex*>(comp_array), 
                reinterpret_cast<fftwf_complex*>(real_array), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        plan_forward_dealias = fftwf_plan_dft_1d(sx, 
                reinterpret_cast<fftwf_complex*>(real_dealiased), 
                reinterpret_cast<fftwf_complex*>(comp_dealiased), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward_dealias = fftwf_plan_dft_1d(sx, 
                reinterpret_cast<fftwf_complex*>(comp_dealiased), 
                reinterpret_cast<fftwf_complex*>(real_dealiased), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        noise_plan = fftwf_plan_dft_1d(sx,
                reinterpret_cast<fftwf_complex*>(noise_gend), 
                reinterpret_cast<fftwf_complex*>(noise_comp), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        cufftPlan1d(&plan_gpu, sx, CUFFT_C2C, 1);
    }
    else // 2d plans
    {
        plan_forward = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(real_array), 
                reinterpret_cast<fftwf_complex*>(comp_array), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(comp_array), 
                reinterpret_cast<fftwf_complex*>(real_array), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        plan_forward_dealias = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(real_dealiased), 
                reinterpret_cast<fftwf_complex*>(comp_dealiased), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward_dealias = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(comp_dealiased), 
                reinterpret_cast<fftwf_complex*>(real_dealiased), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        noise_plan = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(noise_gend), 
                reinterpret_cast<fftwf_complex*>(noise_comp), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        cufftPlan2d(&plan_gpu, sy, sx, CUFFT_C2C);
        // plans take number of rows and then number of columns,
        // so there are y number of rows and x number of columns
    }

    for (int i = 0; i < sx*sy; i++)
    {
        real_array[i].x = 0.0f; real_array[i].y = 0.0f;
        comp_array[i].x = 0.0f; comp_array[i].y = 0.0f;
    }
    cudaMemcpy(real_array_d, real_array, sx * sy * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(comp_array_d, comp_array, sx * sy * sizeof(float2), cudaMemcpyHostToDevice);

    // noise
    rng_d = CURAND_RNG_PSEUDO_PHILOX4_32_10;
    order_d = CURAND_ORDERING_PSEUDO_BEST;
    cudaStreamCreateWithFlags(&stream_d, cudaStreamNonBlocking);
    curandCreateGenerator(&gen_d, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetStream(gen_d, stream_d);
    curandSetGeneratorOffset(gen_d, 0ULL);
    curandSetGeneratorOrdering(gen_d, order_d);
    curandSetPseudoRandomGeneratorSeed(gen_d, time(NULL));

    for (int i = 0; i < sx*sy; i++)
    {
        real_dealiased[i].x = 0.0f; real_dealiased[i].y = 0.0f;
        comp_dealiased[i].x = 0.0f; comp_dealiased[i].y = 0.0f;
    }
    cudaMemcpy(real_dealiased_d, real_dealiased, sx * sy * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(comp_dealiased_d, comp_dealiased, sx * sy * sizeof(float2), cudaMemcpyHostToDevice);

    // boundary conditions
    hasBC = false;
    boundary = NULL;

    // noise
    isNoisy = false;
}

void field::prepareDevice()
{
    // copy implicit terms
    cudaMalloc(reinterpret_cast<void **>(&implicit_terms), implicit.size() * sizeof(pres));
    cudaMemcpy(implicit_terms, &implicit[0], implicit.size() * sizeof(pres), cudaMemcpyHostToDevice);

    // move array of pointers to terms
    terms_h = new float2*[terms.size()];
    cudaMalloc(reinterpret_cast<void **>(&terms_d), terms.size() * sizeof(float2*));
    for (int i = 0; i < terms.size(); i++)
    {
        terms_h[i] = terms[i]->term_comp_d;
    }
    cudaMemcpy(terms_d, terms_h, terms.size() * sizeof(float2*), cudaMemcpyHostToDevice);

    for (int i = 0; i < terms.size(); i++)
    {
        terms[i]->prepareDevice();
    }
}

void field::precalculateImplicit(float dt)
{
    int dimension = 1 + (sy==1 ? 0 : 1);
    for (int j = 0; j < sy; j++)
    {
        for (int i = 0; i < sx; i++)
        {
            int index = j * sx + i;
            float implicitFactor = 0.0f;
            if (dynamic) implicitFactor = 1.0f;
            float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
            float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
            float q2 = qx*qx + qy*qy;
            for (int imp = 0; imp < implicit.size(); imp++)
            {
                float thisImplicit = implicit[imp].preFactor;
                if (implicit[imp].q2n != 0)
                    thisImplicit *= std::pow(q2, implicit[imp].q2n);
                if (implicit[imp].invq !=0)
                {
                    float invq = 0.0f;
                    if (i > 0 || j > 0)
                        invq = 1.0f / std::sqrt(q2);
                    thisImplicit *= std::pow(invq, implicit[imp].invq);
                }
                if (dynamic)
                    implicitFactor -= dt * thisImplicit;
                else
                    implicitFactor += thisImplicit;
            }
            precomp_implicit[index] = implicitFactor;
            // noise
            float noiseFactor = std::sqrt(dt * 2.0f * noise_amplitude.preFactor);
            if (noise_amplitude.q2n != 0)
                noiseFactor *= std::sqrt(std::pow(q2, noise_amplitude.q2n));
            if (noise_amplitude.invq != 0)
            {
                float invq = 0.0f;
                if (index > 0)
                    invq = 1.0f / std::sqrt(q2);
                noiseFactor *= std::sqrt(std::pow(invq, noise_amplitude.invq));
            }
            precomp_noise[index] = noiseFactor;
        }
    }
    cudaMemcpy(precomp_implicit_d, precomp_implicit, sx * sy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(precomp_noise_d, precomp_noise, sx * sy * sizeof(float), cudaMemcpyHostToDevice);
}
