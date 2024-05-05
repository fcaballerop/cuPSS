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
#include "../inc/cupss.h"

void field::common_constructor()
{
    rng.seed(time(NULL));

    real_array = new float2[sx*sy*sz];
    comp_array = new float2[sx*sy*sz];
    cudaMalloc(reinterpret_cast<void **>(&real_array_d), sx * sy * sz * sizeof(float2));
    cudaMalloc(reinterpret_cast<void **>(&comp_array_d), sx * sy * sz * sizeof(float2));

    precomp_implicit = new float[sx*sy*sz];
    cudaMalloc(reinterpret_cast<void **>(&precomp_implicit_d), sx * sy * sz * sizeof(float));

    precomp_noise = new float[sx*sy*sz];
    cudaMalloc(reinterpret_cast<void **>(&precomp_noise_d), sx * sy * sz * sizeof(float));

    noise_comp = new float2[sx*sy*sz];
    noise_gend = new float2[sx*sy*sz];
    cudaMalloc(reinterpret_cast<void **>(&noise_comp_d_r), sx * sy * sz * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&noise_comp_d_i), sx * sy * sz * sizeof(float));

    cudaMalloc(reinterpret_cast<void **>(&gen_noise), sx * sy * sz * sizeof(float));
    cudaMalloc(reinterpret_cast<void **>(&noise_real), sx * sy * sz * sizeof(float2));
    cudaMalloc(reinterpret_cast<void **>(&noise_fourier), sx * sy * sz * sizeof(float2));

    needsaliasing = false;
    aliasing_order = 1;
    comp_dealiased = new float2[sx*sy*sz];
    cudaMalloc(reinterpret_cast<void **>(&comp_dealiased_d), sx * sy * sz * sizeof(float2));
    real_dealiased = new float2[sx*sy*sz];
    cudaMalloc(reinterpret_cast<void **>(&real_dealiased_d), sx * sy * sz * sizeof(float2));

    outputToFile = false;

    // Initialize fftw plans
    if (sz == 1)
    {
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
    }
    else 
    {  // 3d plans
        plan_forward = fftwf_plan_dft_3d(sz, sy, sx,
                reinterpret_cast<fftwf_complex*>(real_array), 
                reinterpret_cast<fftwf_complex*>(comp_array), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftwf_plan_dft_3d(sz, sy, sx,
                reinterpret_cast<fftwf_complex*>(comp_array), 
                reinterpret_cast<fftwf_complex*>(real_array), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        plan_forward_dealias = fftwf_plan_dft_3d(sz, sy, sx,
                reinterpret_cast<fftwf_complex*>(real_dealiased), 
                reinterpret_cast<fftwf_complex*>(comp_dealiased), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward_dealias = fftwf_plan_dft_3d(sz, sy, sx,
                reinterpret_cast<fftwf_complex*>(comp_dealiased), 
                reinterpret_cast<fftwf_complex*>(real_dealiased), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        noise_plan = fftwf_plan_dft_3d(sz, sy, sx,
                reinterpret_cast<fftwf_complex*>(noise_gend), 
                reinterpret_cast<fftwf_complex*>(noise_comp), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        cufftPlan3d(&plan_gpu, sz, sy, sx, CUFFT_C2C);
    }

    for (int i = 0; i < sx*sy*sz; i++)
    {
        real_array[i].x = 0.0f; real_array[i].y = 0.0f;
        comp_array[i].x = 0.0f; comp_array[i].y = 0.0f;
    }
    cudaMemcpy(real_array_d, real_array, sx * sy * sz * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(comp_array_d, comp_array, sx * sy * sz * sizeof(float2), cudaMemcpyHostToDevice);

    // noise
    rng_d = CURAND_RNG_PSEUDO_PHILOX4_32_10;
    order_d = CURAND_ORDERING_PSEUDO_BEST;
    cudaStreamCreateWithFlags(&stream_d, cudaStreamNonBlocking);
    curandCreateGenerator(&gen_d, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetStream(gen_d, stream_d);
    curandSetGeneratorOffset(gen_d, 0ULL);
    curandSetGeneratorOrdering(gen_d, order_d);
    curandSetPseudoRandomGeneratorSeed(gen_d, time(NULL));

    for (int i = 0; i < sx*sy*sz; i++)
    {
        real_dealiased[i].x = 0.0f; real_dealiased[i].y = 0.0f;
        comp_dealiased[i].x = 0.0f; comp_dealiased[i].y = 0.0f;
    }
    cudaMemcpy(real_dealiased_d, real_dealiased, sx * sy * sz * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(comp_dealiased_d, comp_dealiased, sx * sy * sz * sizeof(float2), cudaMemcpyHostToDevice);

    // callback functions
    hasCB = false;
    callback = NULL;
    hasCBFourier = false;
    callbackFourier = NULL;

    // noise
    isNoisy = false;
}

field::field(int _sx, float _dx) : sx(_sx), sy(1), sz(1), dx(_dx), dy(1), dz(1), stepqx(2.0f*PI/(_dx * (float)_sx)), stepqy(2.0f*PI/(1.0f * (float)1)), stepqz(2.0f*PI/(1.0f * (float)1)), rng(rd()), dist(std::normal_distribution<>(0, 1.0))
{
    common_constructor();
}

field::field(int _sx, int _sy, float _dx, float _dy) : sx(_sx), sy(_sy), sz(1), dx(_dx), dy(_dy), dz(1.0f), stepqx(2.0f*PI/(_dx * (float)_sx)), stepqy(2.0f*PI/(_dy * (float)_sy)), stepqz(2.0f*PI/(1.0f * (float)1)), rng(rd()), dist(std::normal_distribution<>(0, 1.0))
{
    common_constructor();
}

field::field(int _sx, int _sy, int _sz, float _dx, float _dy, float _dz) : sx(_sx), sy(_sy), sz(_sz), dx(_dx), dy(_dy), dz(_dz), stepqx(2.0f*PI/(_dx * (float)_sx)), stepqy(2.0f*PI/(_dy * (float)_sy)), stepqz(2.0f*PI/(_dz * (float)_sz)), rng(rd()), dist(std::normal_distribution<>(0, 1.0))
{
    common_constructor();
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
    // int dimension = 1 + (sy==1 ? 0 : 1);
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                float implicitFactor = 0.0f;
                if (dynamic) implicitFactor = 1.0f;
                float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
                float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
                float qz = (k < (sz+1)/2 ? (float)k : (float)(k - sz)) * stepqz;
                float q2 = qx*qx + qy*qy + qz*qz;
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
    }
    cudaMemcpy(precomp_implicit_d, precomp_implicit, sx * sy * sz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(precomp_noise_d, precomp_noise, sx * sy * sz * sizeof(float), cudaMemcpyHostToDevice);
}
