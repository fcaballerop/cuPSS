#include <cmath>
#include <cuda_device_runtime_api.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>
#include <fftw3.h>
#include "../inc/cupss.h"

term::term(int _sx, int _sy, float _dx, float _dy) : sx(_sx), sy(_sy), dx(_dx), dy(_dy), stepqx(2.0f*PI/(_dx * (float)_sx)), stepqy(2.0f*PI/(_dy * (float)_sy))
{
    // sx = _sx;
    // sy = _sy;
    // dx = _dx;
    // dy = _dy;
    // stepqx = 2.0f * PI / (dx * sx);
    // stepqy = 2.0f * PI / (dy * sy);
    term_real = new float2[sx*sy];
    term_comp = new float2[sx*sy];
    cudaMalloc(reinterpret_cast<void **>(&term_real_d), sx * sy * sizeof(float2));
    cudaMalloc(reinterpret_cast<void **>(&term_comp_d), sx * sy * sizeof(float2));
    isCUDA = true;
    

    // Initialize fftw plans
    if (sy == 1) // 1d plans
    {
        plan_forward = fftwf_plan_dft_1d(sx, 
                reinterpret_cast<fftwf_complex*>(term_real), 
                reinterpret_cast<fftwf_complex*>(term_comp), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftwf_plan_dft_1d(sx, 
                reinterpret_cast<fftwf_complex*>(term_comp), 
                reinterpret_cast<fftwf_complex*>(term_real), 
                FFTW_BACKWARD, FFTW_ESTIMATE);
        cufftPlan1d(&plan_gpu, sx, CUFFT_C2C, 1);
    }
    else // 2d plans
    {
        plan_forward = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(term_real), 
                reinterpret_cast<fftwf_complex*>(term_comp), 
                FFTW_FORWARD, FFTW_ESTIMATE);
        plan_backward = fftwf_plan_dft_2d(sy, sx,
                reinterpret_cast<fftwf_complex*>(term_comp), 
                reinterpret_cast<fftwf_complex*>(term_real), 
                FFTW_BACKWARD, FFTW_ESTIMATE);

        cufftPlan2d(&plan_gpu, sy, sx, CUFFT_C2C);
    }

    // initialize arrays to 0, avoids problems
    // if updates are called before filling in values
    for (int i = 0; i < sx*sy; i++)
    {
        term_real[i].x = 0.0f; term_real[i].y = 0.0f;
        term_comp[i].x = 0.0f; term_comp[i].y = 0.0f;
    }
    cudaMemcpy(term_real_d, term_real, sx * sy * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(term_comp_d, term_comp, sx * sy * sizeof(float2), cudaMemcpyHostToDevice);

    precomp_prefactor_h = new float[sx * sy];
    cudaMalloc(reinterpret_cast<void **>(&precomp_prefactor_d), sx * sy * sizeof(float));
}

int term::prepareDevice()
{
    cudaMalloc(reinterpret_cast<void **>(&prefactors_d), prefactors_h.size() * sizeof(pres));
    cudaMemcpy(prefactors_d, &prefactors_h[0], prefactors_h.size() * sizeof(pres), cudaMemcpyHostToDevice);

    product_h = new float2*[product.size()];
    cudaMalloc(reinterpret_cast<void **>(&product_d), product.size() * sizeof(float2*));
    for (int i = 0; i < product.size(); i++)
    {
        if (product.size() == 1)
        {
            product_h[i] = product[i]->real_array_d;
        }
        else 
        {
            product[i]->needsaliasing = true;
            if (product[i]->aliasing_order < product.size())
                product[i]->aliasing_order = product.size();
            product_h[i] = product[i]->real_dealiased_d;
        }
    }
    cudaMemcpy(product_d, product_h, product.size() * sizeof(float2*), cudaMemcpyHostToDevice);

    precomputePrefactors();
    return 0;
}

int term::precomputePrefactors()
{
    for (int j = 0; j < sy; j++)
    {
        for (int i = 0; i < sx; i++)
        {
            int index = j * sx + i;
            float totalPrefactor = 0.0f;
            int multiply_by_i = (prefactors_h[0].iqx + prefactors_h[0].iqy) % 2;
            for (int p = 0; p < prefactors_h.size(); p++)
            {
                float num_prefactor = prefactors_h[p].preFactor;
                int imaginary_units = prefactors_h[p].iqx + prefactors_h[p].iqy;
                int multiply_by_i_this = imaginary_units % 2;
                if (multiply_by_i != multiply_by_i_this)
                {
                    std::cout << "PANIC: Inconsistent powers of I in prefactors" << std::endl;
                }
                int negate = -2 * (((imaginary_units - multiply_by_i)/2)%2) + 1;
                num_prefactor *= (float)negate;
                float qx = (i < (sx+1)/2 ? (float)i : (float)(i - sx)) * stepqx;
                float qy = (j < (sy+1)/2 ? (float)j : (float)(j - sy)) * stepqy;
                float q2 = qx*qx + qy*qy;
                if (prefactors_h[p].q2n > 0)
                {
                    num_prefactor *= std::pow(q2, prefactors_h[p].q2n);    
                }
                if (prefactors_h[p].iqx > 0)
                {
                    num_prefactor *= std::pow(qx, prefactors_h[p].iqx);    
                }
                if (prefactors_h[p].iqy > 0)
                {
                    num_prefactor *= std::pow(qy, prefactors_h[p].iqy);    
                }
                if (prefactors_h[p].invq > 0)
                {
                    float invq = 0.0f;
                    if (index > 0)
                        invq = 1.0f / std::sqrt(q2);
                    num_prefactor *= std::pow(invq, prefactors_h[p].invq);    
                }
                totalPrefactor += num_prefactor;
            }
            precomp_prefactor_h[index] = totalPrefactor;
            if (multiply_by_i)
            {
                multiply_by_i_pre = 1;
            }
            else {
                multiply_by_i_pre = 0;
            }
        }
    }

    cudaMemcpy(precomp_prefactor_d, precomp_prefactor_h, sx * sy * sizeof(float), cudaMemcpyHostToDevice);
    return 0;
}
