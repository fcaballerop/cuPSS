#include <cmath>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <fftw3.h>
#include "../inc/cupss.h"
#include "../inc/cupss/term_kernels.cuh"

int term::update()
{
    // There should be a check. If the term is made of only one field, dont compute a product
    // and trasnform to comp, but set comp directly.
    if (product.size() == 1)
        copyComp();
    else
    {
        computeProduct();
        toComp();
    }
    applyPres_vector();
    return 0;
}

int term::setPrefactorString(const std::vector<std::string> &_these_terms)
{
    for (int i = 0; i < _these_terms.size(); i++)
    {
        prefactor_strings.push_back(_these_terms[i]);
    }
    return 0;
}

void term::printPrefactorString()
{
    for (int i = 0; i < prefactor_strings.size(); i++)
    {
        std::cout << prefactor_strings[i] << std::endl;
    }
    for (const auto &x : usedParameters)
    {
        std::cout << x.first << " " << x.second << std::endl;
    }
}

int term::toComp()
{
    // Transform the real vector to Fourier space
    // Should split into cuFFT or fftw3 transforms
    // 
    // For now just a temporary direct copy
    if (isCUDA)
    {
        cudaDeviceSynchronize();
        cufftExecC2C(plan_gpu, term_real_d, term_comp_d, CUFFT_FORWARD);
    }
    else 
    {
        fftwf_execute(plan_forward);
    }
    return 0;
}

int term::computeProduct()
{
    if (isCUDA)
    {
        cudaDeviceSynchronize();
        computeProduct_gpu(product_d, term_real_d, product.size(), sx, sy, sz, blocks, threads_per_block);
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
                    float result = 1.0f;
                    if (product.size() == 1)
                    {
                        // copy from real
                        // never executed, if size is 1 comp is copied directly
                    }
                    else 
                    {
                        for (int term = 0; term < product.size(); term++)
                        {
                            result *= product[term]->real_dealiased[index].x;
                        }
                    }
                    term_real[index].x = result;
                    term_real[index].y = 0.0f;
                }
            }
        }
    }
    return 0;
}

int term::copyComp()
{
    if (isCUDA)
    {
        copyComp_gpu(term_comp_d, product[0]->comp_array_d, sx, sy, sz, blocks, threads_per_block);
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
                    term_comp[index].x = product[0]->comp_array[index].x;
                    term_comp[index].y = product[0]->comp_array[index].y;
                }
            }
        }
    }
    return 0;
}

int term::applyPrefactors()
{
    if (isCUDA)
    {
        applyPrefactor_gpu(term_comp_d, prefactors.preFactor, prefactors.q2n, prefactors.iqx, prefactors.iqy, prefactors.iqz, prefactors.invq, sx, sy, sz, stepqx, stepqy, stepqz, blocks, threads_per_block);
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
                    term_comp[index].x *= prefactors.preFactor;
                    term_comp[index].y *= prefactors.preFactor;
                    if (prefactors.q2n | prefactors.iqx | prefactors.iqy | prefactors.iqz | prefactors.invq)     // If any of them are positive
                    {
                        float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
                        float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
                        float qz = (k < (sz+1)/2 ? (float)k : (float)(k - sz)) * stepqz;
                        float q2 = qx*qx + qy*qy + qz*qz;
                        if (prefactors.q2n > 0)
                        {
                            float powerq2n = std::pow(q2, prefactors.q2n);
                            term_comp[index].x *= powerq2n;
                            term_comp[index].y *= powerq2n;
                        }
                        if (prefactors.iqx > 0)
                        {
                            // Calculates (iqx)^n = i^n2 (-1)^a (qx)^n
                            // where a = (n-n%2)/2, and n2 = n%2
                            int a = (prefactors.iqx - prefactors.iqx % 2) / 2;
                            int n2 = prefactors.iqx % 2;
                            int m1power = 2 * (- a % 2) + 1; // = (-1)^a, save one power calculation
                            float qpower = qx;
                            if (prefactors.iqx > 1) qpower = std::pow(qx, prefactors.iqx);
                            term_comp[index].x *= (float)m1power * qpower;
                            term_comp[index].y *= (float)m1power * qpower;
                            if (n2 == 1)    // if odd power of iqx, conjugate.
                            {
                                float auxiliar = term_comp[index].x;
                                term_comp[index].x = - term_comp[index].y;
                                term_comp[index].y = auxiliar;
                            }
                        }
                        if (prefactors.iqy > 0)
                        {
                            // Same as qx
                            int a = (prefactors.iqy - prefactors.iqy % 2) / 2;
                            int n2 = prefactors.iqy % 2;
                            int m1power = 2 * (- a % 2) + 1; // = (-1)^a, save one power calculation
                            float qpower = qy;
                            if (prefactors.iqy > 1) qpower = std::pow(qy, prefactors.iqy);
                            term_comp[index].x *= (float)m1power * qpower;
                            term_comp[index].y *= (float)m1power * qpower;
                            if (n2 == 1)    // if odd power of iqx, conjugate.
                            {
                                float auxiliar = term_comp[index].x;
                                term_comp[index].x = - term_comp[index].y;
                                term_comp[index].y = auxiliar;
                            }
                        }
                        if (prefactors.iqz > 0)
                        {
                            // Same as qx
                            int a = (prefactors.iqz - prefactors.iqz % 2) / 2;
                            int n2 = prefactors.iqz % 2;
                            int m1power = 2 * (- a % 2) + 1; // = (-1)^a, save one power calculation
                            float qpower = qz;
                            if (prefactors.iqz > 1) qpower = std::pow(qz, prefactors.iqz);
                            term_comp[index].x *= (float)m1power * qpower;
                            term_comp[index].y *= (float)m1power * qpower;
                            if (n2 == 1)    // if odd power of iqx, conjugate.
                            {
                                float auxiliar = term_comp[index].x;
                                term_comp[index].x = - term_comp[index].y;
                                term_comp[index].y = auxiliar;
                            }
                        }
                        if (prefactors.invq > 0)
                        {
                            float invq = 0.0f;
                            if (i > 0 || j > 0)
                                invq = 1.0f / std::pow(std::abs(std::sqrt(q2)), prefactors.invq);
                            term_comp[index].x *= invq;
                            term_comp[index].y *= invq;
                        }
                    }
                }
            }
        }
    }
    return 0;
}

int term::applyPres_vector()
{
    if (isCUDA)
    {
        // applyPres_vector_gpu(term_comp_d, prefactors_d, prefactors_h.size(), sx, sy, stepqx, stepqy);
        cudaDeviceSynchronize();
        applyPres_vector_pre_gpu(term_comp_d, precomp_prefactor_d, multiply_by_i_pre, sx, sy, sz, blocks, threads_per_block);
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
                    term_comp[index].x *= precomp_prefactor_h[index];
                    term_comp[index].y *= precomp_prefactor_h[index];
                    if (multiply_by_i_pre)
                    {
                        float aux = term_comp[index].x;
                        term_comp[index].x =  - term_comp[index].y;
                        term_comp[index].y = aux;
                    }
                }
            }
        }
    }
    return 0;
}

