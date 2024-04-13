#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
#include "../inc/defines.h"
#include "../inc/evolver.h"
#include "../inc/field.h"
#include "../inc/term.h"

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 128
#define NY 128

void createIntegral(float2 *, float2 *);
__global__ void createIntegral_k(float2 *, float2 *);

int main(int argc, char **argv)
{
    evolver system(1, NX, NY, 1.0f, 1.0f, 0.1f, 100);

    system.createField("phi", true);        // 0
    system.createField("phisq", false);     // 1
    system.createField("intphisq", false);  // 2
    system.createField("iqxphi", false);    // 3
    system.createField("iqyphi", false);    // 4

    system.addParameter("k", 4.0f);
    system.addParameter("lambda", 0.5f);
    system.addParameter("pi", 3.141592f);
    system.addParameter("Rt", 20.0f);

    system.addParameter("ep", 1.0f);

    system.addParameter("Mphi", 1.0f);

    system.addEquation("dt phi + (1+k*q^2)*phi = +3*phi^2-2*phi^3 +4*lambda*phi- 4*lambda*1/pi*1/Rt*1/Rt *phi * intphisq - 2*ep*cp^2*phi");

    system.addEquation("phisq = phi^2");
    system.addEquation("iqxphi = iqx*phi");
    system.addEquation("iqyphi = iqy*phi");

    int Rt = 80;
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            if ((i-NX/2)*(i-NX/2) + (j-NY/2)*(j-NY/2) < Rt*Rt)
                system.fields[0]->real_array[index].x = 1.0f;
            else
                system.fields[0]->real_array[index].x = 0.0f;
        }
    }

    
    system.prepareProblem();

    int steps = 100000;
    int check = steps/100;
    if (check < 1) check = 1;
    
    system.printInformation();

    float2 *phiFourier = new float2[NX*NY];
    float *amplitudeHistogram = new float[NX > NY ? NX : NY];
    float *amplitudeCounts = new float[NX > NY ? NX : NY];
    for (int i = 0; i < (NX > NY ? NX : NY); i++)
    {
        amplitudeHistogram[i] = 0.0f;
        amplitudeCounts[i] = 0.0f;
    }

    for (int i = 0; i < steps; i++)
    {
        system.advanceTime();
        createIntegral(system.fields[1]->comp_array_d, system.fields[2]->real_array_d);
        system.fields[2]->toComp();
        if (i % check == 0)
        {
            cudaMemcpy(phiFourier, system.fieldsMap["phi"]->comp_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
            for (int x = 0; x < NX; x++)
            {
                for (int y = 0; y < NY; y++)
                {
                    int xx = x < NX/2 ? x : NX - x;
                    int yy = y < NY/2 ? y : NY - y;
                    int index = (int)std::floor(std::sqrt((float)(xx*xx) + (float)(yy*yy)));
                    amplitudeHistogram[index] += std::sqrt(phiFourier[x*NY+y].x*phiFourier[x*NY+y].x + phiFourier[x*NY+y].y*phiFourier[x*NY+y].y);
                    amplitudeCounts[index] += 1.0f;
                }
            }
            std::string histFilename = "data/histogram.out." + std::to_string(i);
            FILE *fp = fopen(histFilename.c_str(), "w+");
            for (int k = 0; k < (NX > NY ? NX : NY); k++)
            {
                amplitudeHistogram[k] /= amplitudeCounts[k];
                fprintf(fp, "%f\n", amplitudeHistogram[k]);
            }
            fclose(fp);
            float max = 0.0f;
            int maxInd = 0;
            for (int k = 0; k < (NX > NY ? NX : NY); k++)
            {
                if (amplitudeHistogram[k] > max)
                {
                    max = amplitudeHistogram[k];
                    maxInd = k;
                }
            }
            std::cout << maxInd << std::endl;
            // std::cout << "Progress: " << i/check << "%\r";
            // std::cout.flush();
        }
    }

    return 0;
}

void createIntegral(float2 *source, float2 *output)
{
    dim3 TPB(32,32);
    dim3 blocks(NX/32, NY/32);

    createIntegral_k<<<blocks,TPB>>>(source, output);
}

__global__ void createIntegral_k(float2 *source, float2 *output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int index = j*NX+i;

    if (index < NX*NY)
    {
        output[index].x = source[0].x;
    }
}
