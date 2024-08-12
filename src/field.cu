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
#include "field.h"

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
                    comp_array_d, sx, sy, sz, stepqx, stepqy, stepqz, precomp_implicit_d, isNoisy, noise_fourier, precomp_noise_d, dt, blocks, threads_per_block);
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
            setNotDynamic(dt);
        }
        else
        {
            setDynamic(dt);
        }
    }
    
    if (hasCBFourier)
    {
        if (callbackFourier == NULL)
            std::cout << "Wants to apply callback function in Fourier space but pointer to function is NULL" << std::endl;
        else {
            if ( isCUDA )
                callbackFourier(system_p, comp_array_d, sx, sy, sz);
            else 
                callbackFourier(system_p, comp_array, sx, sy, sz);
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

    // apply boundary conditions
    for (std::map<int,BoundaryCondition>::iterator bc = boundary_conditions.begin(); bc!=boundary_conditions.end(); bc++)
    {
        BoundaryCondition& boundary = ((*bc).second);
        if (isCUDA)
        {
            boundary(real_array_d);
            if (needsaliasing)
            {
                boundary(real_dealiased_d);
            }
        }
        else {
            boundary(real_array);
            if (needsaliasing)
            {
                boundary(real_dealiased);
            }
        }
    }
    // If there are callback functions defined for the field, they should be executed
    // here, maybe take a pointer to a function to be called back here
    // do BC here


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

void field::setNotDynamic(float dt)
{
    float sdt = 0.0f;
    if (isNoisy)
        sdt = 1.0f / std::sqrt(dt);
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
                if (isNoisy)
                {
                    if (terms.size() == 0)
                    {
                        comp_array[index].x = sdt * precomp_noise[index] * noise_comp[index].x;
                        comp_array[index].y = sdt * precomp_noise[index] * noise_comp[index].y;
                    }
                    else 
                    {
                        comp_array[index].x += sdt * precomp_noise[index] * noise_comp[index].x;
                        comp_array[index].y += sdt * precomp_noise[index] * noise_comp[index].y;
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
                if (isNoisy)
                {
                    comp_array[index].x += precomp_noise[index] * noise_comp[index].x;
                    comp_array[index].y += precomp_noise[index] * noise_comp[index].y;
                }
                // After adding all explicit terms, we divide over the implicits
                if (implicit.size() > 0)
                {
                    comp_array[index].x /= precomp_implicit[index];
                    comp_array[index].y /= precomp_implicit[index];
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
        default:
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

void field::copyHostToDevice()
{
    cudaMemcpy(real_array_d, real_array, sx*sy*sz*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(comp_array_d, comp_array, sx*sy*sz*sizeof(float2), cudaMemcpyHostToDevice);
}

void field::copyDeviceToHost()
{
    cudaMemcpy(real_array, real_array_d, sx*sy*sz*sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(comp_array, comp_array_d, sx*sy*sz*sizeof(float2), cudaMemcpyDeviceToHost);
}

void field::copyRealHostToDevice()
{
    cudaMemcpy(real_array_d, real_array, sx*sy*sz*sizeof(float2), cudaMemcpyHostToDevice);
}

void field::copyRealDeviceToHost()
{
    cudaMemcpy(real_array, real_array_d, sx*sy*sz*sizeof(float2), cudaMemcpyDeviceToHost);
}

void field::writeToFile(int currentTimeStep, int dim, int writePrecision)
{
    if (!outputToFile)
        return;

    if (isCUDA)
        copyRealDeviceToHost();

    FILE *fp;
    std::string fileName = "data/" + name + ".csv." + std::to_string(currentTimeStep);
    
    fp = fopen(fileName.c_str(), "w+");
    if (fp == NULL)
    {
        std::cout << "Error creating output file at timestep" << currentTimeStep << std::endl;
        std::exit(1);
    }

    std::string outFormat = "%i, ";
    fprintf(fp, "x, ");
    if (dim == 2)
    {
        fprintf(fp, "y, ");
        outFormat += "%i, ";
    }
    if (dim == 3)
    {
        fprintf(fp, "y, z, ");
        outFormat += "%i, %i, ";
    }
    fprintf(fp, "%s\n", name.c_str());
    outFormat += "%." + std::to_string(writePrecision) + "f\n";

    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                int bytesWritten = 0;
                if (std::isnan(real_array[index].x))
                {
                    std::cout << "NaN found in field " << name << std::endl;
                    std::exit(1);
                }
                if (dim == 1)
                    bytesWritten = fprintf(fp, outFormat.c_str(), i, real_array[index].x);
                if (dim == 2)
                    bytesWritten = fprintf(fp, outFormat.c_str(), i, j, real_array[index].x);
                if (dim == 3)
                    bytesWritten = fprintf(fp, outFormat.c_str(), i, j, k, real_array[index].x);
                if (bytesWritten < 0)
                {
                    std::cout << "Error writing data to output file " << fileName << std::endl;
                    std::exit(1);
                }
            }
        }
    }
    fclose(fp);
}

float field::getStepqx()
{
    return stepqx;
}
float field::getStepqy()
{
    return stepqy;
}
float field::getStepqz()
{
    return stepqz;
}

void field::addBoundaryCondition(BoundaryCondition BC)
{
    boundary_conditions[BC.get_direction()]=BC;
    boundary_conditions[BC.get_direction()].initalize(this);

int field::addImplicitString(const std::string &_this_term)
{
    implicit_prefactor_strings.push_back(_this_term);
    return 0;
}

void field::printImplicitString()
{
    for (int i = 0; i < implicit_prefactor_strings.size(); i++)
    {
        std::cout << implicit_prefactor_strings[i] << std::endl;
    }
    for (const auto &x : usedParameters)
    {
        std::cout << x.first << " " << x.second << std::endl;
    }
}

int field::updateParameter(const std::string &name, float value)
{
    int implicitsChanged = usedParameters[name];
    
    if (implicitsChanged)
    {
        int dyn = 0;
        if (dynamic) dyn = 1;
        system_p->_parser->recalculateImplicits(implicit_prefactor_strings, implicit, dyn);
        precalculateImplicit(system_p->dt);
    }

    for (int i = 0; i < terms.size(); i++)
    {
        int termChanged = terms[i]->usedParameters[name];

        if (termChanged)
        {
            system_p->_parser->recalculateImplicits(terms[i]->prefactor_strings, terms[i]->prefactors_h, 0);
            terms[i]->precomputePrefactors();
        }
    }
    return 0;
>>>>>>> 58484fa0cca89ba0f4f96c0ea11ac22cb2e3fec4
}
