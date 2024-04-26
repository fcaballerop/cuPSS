#include <cmath>
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
#include "../inc/cupss/field_kernels.cuh"

int field::updateTerms()
{
    if (isNoisy)
    {
        createNoise();
    }
    for (int i = 0; i < terms.size(); i++)
    {
        terms[i]->update();
    }
    return 0;
}

int field::setRHS(float dt)
{
    // Now each term is calculated and ready in Fourier space
    // The RHS is just the sum of terms
    if (isCUDA)
    {
        if (!dynamic)
        {
            setNotDynamic_gpu(terms_d, terms.size(), implicit_terms, implicit.size(), 
                    comp_array_d, sx, sy, sz, stepqx, stepqy, stepqz, precomp_implicit_d, blocks, threads_per_block);
        }
        else 
        {
            setDynamic_gpu(terms_d, terms.size(), implicit_terms, implicit.size(),
                    comp_array_d, sx, sy, sz, stepqx, stepqy, stepqz, dt, precomp_implicit_d,
                    isNoisy, noise_fourier, precomp_noise_d, blocks, threads_per_block);
        }
    }
    else 
    {
        if (!dynamic)
        {
            setNotDynamic();
        }
        else
        {
            setDynamic(dt);
        }
    }
    
    // dealiasing
    if (needsaliasing)
    {
        dealias();
    }
    // Transform to Real space
    toReal();
    // Normalize and remove imaginary errors
    normalize();

    // If there are callback functions defined for the field, they should be executed
    // here, maybe take a pointer to a function to be called back here
    if (hasCB)
    {
        if (callback == NULL)
        {
            std::cout << "Wants to apply callback function but pointer to function is NULL" << std::endl;
        }
        else {
            if ( isCUDA )
            {
                callback(system_p, real_array_d, sx, sy, sz);
                if (needsaliasing)
                    callback(system_p, real_dealiased_d, sx, sy, sz);
            }
            else {
                callback(system_p, real_array, sx, sy, sz);
                if (needsaliasing)
                    callback(system_p, real_dealiased, sx, sy, sz);
            }
        }
    }

    // Transform back
    toComp();
    
    return 0;
}

void field::setNotDynamic()
{
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                for (int term = 0; term < terms.size(); term++)
                {
                    if (term == 0)
                    {
                        comp_array[index].x = terms[term]->term_comp[index].x;
                        comp_array[index].y = terms[term]->term_comp[index].y;
                    }
                    else
                    {
                        comp_array[index].x += terms[term]->term_comp[index].x;
                        comp_array[index].y += terms[term]->term_comp[index].y;
                    }
                }
                if (implicit.size() > 0 && index != 0) // last condition easy fix for q=0 case
                {
                    float implicitFactor = 0.0f;
                    float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
                    float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
                    float qz = (k < (sz+1)/2 ? (float)k : (float)(k - sz)) * stepqz;
                    float q2 = qx*qx + qy*qy + qz*qz;
                    for (int imp = 0; imp < implicit.size(); k++)
                    {
                        float thisImplicit = implicit[imp].preFactor;
                        // At this point only scalars allowed (q2n and invq)
                        if (implicit[imp].q2n != 0)
                            thisImplicit *= std::pow(q2, implicit[imp].q2n);
                        if (implicit[imp].invq != 0)
                        {
                            float invq = 0.0f;
                            if (i > 0 || j > 0)
                                invq = 1.0f / std::sqrt(q2);
                            thisImplicit *= std::pow(invq, implicit[imp].invq);
                        }
                        implicitFactor += thisImplicit;
                    }
                    comp_array[index].x /= implicitFactor;
                    comp_array[index].y /= implicitFactor;
                }
            }
        }
    }
}


void field::setDynamic(float dt)
{
    switch (integrator) 
    {
        case EULER:
            stepEuler(dt);
            break;
        case RK2:
            stepRK2(dt);
            break;
        case RK4:
            stepRK4(dt);
            break;
        default:
            stepEuler(dt);
            break;
    }
}



void field::stepEuler(float dt)
{
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                for (int term = 0; term < terms.size(); term++)
                {
                    comp_array[index].x += dt * terms[term]->term_comp[index].x;
                    comp_array[index].y += dt * terms[term]->term_comp[index].y;
                }
                // After adding all explicit terms, we divide over the implicits
                if (implicit.size() > 0)
                {
                    comp_array[index].x /= precomp_implicit[index];
                    comp_array[index].y /= precomp_implicit[index];
                }
                if (isNoisy)
                {
                    comp_array[index].x += precomp_noise[index] * noise_comp[index].x;
                    comp_array[index].y += precomp_noise[index] * noise_comp[index].y;
                }
            }
        }
    }
}

void field::stepRK2(float dt)
{
    std::cout << "RK2 not implemented" << std::endl;
}

void field::stepRK4(float dt)
{
    std::cout << "RK4 not implemented" << std::endl;
}

void field::dealias()
{
    if (isCUDA)
    {
        cudaDeviceSynchronize();
        dealias_gpu(comp_array_d, comp_dealiased_d, sx, sy, sz, aliasing_order, blocks, threads_per_block);
    }
    else {
        for (int k = 0; k < sz; k++)
        {
            for (int j = 0; j < sy; j++)
            {
                for (int i = 0; i < sx; i++)
                {
                    int index = k * sx * sy + j * sx + i;
                    int ni = i;
                    int nj = j;
                    int nk = k;
                    if (ni > sx/2) ni -= sx;
                    if (nj > sy/2) nj -= sy;
                    if (nk > sz/2) nk -= sz;
                    if (std::abs(ni) > sx/(aliasing_order+1) || std::abs(nj) > sy/(aliasing_order+1) || std::abs(nj) > sz/(aliasing_order+1))
                    {
                        comp_dealiased[index].x = 0.0f;
                        comp_dealiased[index].y = 0.0f;
                    }
                    else {
                        comp_dealiased[index].x = comp_array[index].x;
                        comp_dealiased[index].y = comp_array[index].y;
                    }
                }
            }
        }
    }
}

void field::setToZero()
{
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                comp_array[index].x = 0.0f;
                comp_array[index].y = 0.0f;
            }
        }
    }
}

void field::toReal()
{
    if (isCUDA) {
        cudaDeviceSynchronize();
        cufftExecC2C(plan_gpu, comp_array_d, real_array_d, CUFFT_INVERSE);
        if (needsaliasing)
            cufftExecC2C(plan_gpu, comp_dealiased_d, real_dealiased_d, CUFFT_INVERSE);
    }
    else 
    {
        fftwf_execute(plan_backward);
        if (needsaliasing)
            fftwf_execute(plan_backward_dealias);
    }
}

void field::toComp()
{
    if (isCUDA)
    {
        cudaDeviceSynchronize();
        cufftExecC2C(plan_gpu, (cufftComplex *)real_array_d, (cufftComplex *)comp_array_d, CUFFT_FORWARD);
        if (needsaliasing)  // Not needed, comp dealiased are never used in the fields themselves, only in terms
            cufftExecC2C(plan_gpu, real_dealiased_d, comp_dealiased_d, CUFFT_FORWARD);
    }
    else
    {
        fftwf_execute(plan_forward);
        if (needsaliasing)
            fftwf_execute(plan_forward_dealias);
    }
}

void field::normalize()
{
    if (isCUDA)
    {
        cudaDeviceSynchronize();
        normalize_gpu(real_array_d, sx, sy, sz, blocks, threads_per_block);
        if (needsaliasing)
            normalize_gpu(real_dealiased_d, sx, sy, sz, blocks, threads_per_block);
    }
    else
    {
        float normalization = 1.0f / ((float)(sx*sy*sz));
        for (int k = 0; k < sz; k++)
        {
            for (int j = 0; j < sy; j++)
            {
                for (int i = 0; i < sx; i++)
                {
                    int index = k * sx * sy + j * sx + i;
                    real_array[index].x *= normalization;
                    real_array[index].y = 0.0f;
                    if (needsaliasing)
                    {
                        real_dealiased[index].x *= normalization;
                        real_dealiased[index].y = 0.0f;
                    }
                }
            }
        }
    }
}

void field::createNoise()
{
    if (isCUDA)
    {
        switch (noiseType)
        {
            case GaussianWhite:
            // curandGenerateNormal(gen_d, noise_comp_d_r, sx*sy, 0.0f, 0.707f); // 1/sqrt(2)
            // cudaDeviceSynchronize();
            // curandGenerateNormal(gen_d, noise_comp_d_i, sx*sy, 0.0f, 0.707f);
            curandGenerateNormal(gen_d, gen_noise, sx*sy*sz, 0.0f, 1.0f);
            cudaDeviceSynchronize();
            copyToFloat2_gpu(gen_noise, noise_real, sx, sy, sz, blocks, threads_per_block);
            cudaDeviceSynchronize();
            cufftExecC2C(plan_gpu, noise_real, noise_fourier, CUFFT_FORWARD);
            // correctNoiseAmplitude_gpu(noise_fourier, precomp_noise_d, sx, sy);
            // createNoise_gpu(noise_comp_d_r, noise_comp_d_i, sx, sy, precomp_noise_d);
        break;
        }
    }
    else
    {
        for (int k = 0; k < sz; k++)
        {
            for (int j = 0; j < sy; j++)
            {
                for (int i = 0; i < sx; i++)
                {
                    int index = k * sx * sy + j * sx + i;
                    float r1 = dist(rng);
                    noise_gend[index].x = r1;
                    noise_gend[index].y = 0.0f;
                }
            }
        }
        fftwf_execute(noise_plan);
    }
}
