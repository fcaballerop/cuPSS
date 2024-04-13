#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
#include <chrono>
#include <ctime>
#include "../inc/defines.h"
#include "../inc/evolver.h"
#include "../inc/field.h"
#include "../inc/term.h"
#include "../inc/parser.h"

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 64
#define NY 64
#define NSTEPS 100

int main(int argc, char **argv)
{
    evolver system1d_cpu(0, NX, 1, 1.0f, 1.0f, 1.0f, 1);
    evolver system2d_cpu(0, NX, NY, 1.0f, 1.0f, 1.0f, 1);
    evolver system1d_gpu(1, NX, 1, 1.0f, 1.0f, 1.0f, 1);
    evolver system2d_gpu(1, NX, NY, 1.0f, 1.0f, 1.0f, 1);

    system1d_cpu.createField("init1dcpu", false);
    system1d_cpu.createField("lap1dcpu", false);
    system1d_cpu.createField("iqx1dcpu", false);
    system1d_cpu.createField("invq1dcpu", false);
    system1d_cpu.createField("lapinvq1dcpu", false);

    system1d_gpu.createField("init1dgpu", false);
    system1d_gpu.createField("lap1dgpu", false);
    system1d_gpu.createField("iqx1dgpu", false);
    system1d_gpu.createField("invq1dgpu", false);
    system1d_gpu.createField("lapinvq1dgpu", false);
    
    system2d_cpu.createField("init2dcpu", false);
    system2d_cpu.createField("lap2dcpu", false);
    system2d_cpu.createField("iqx2dcpu", false);
    system2d_cpu.createField("iqy2dcpu", false);
    system2d_cpu.createField("invq2dcpu", false);
    system2d_cpu.createField("lapinvq2dcpu", false);

    system2d_gpu.createField("init2dgpu", false);
    system2d_gpu.createField("lap2dgpu", false);
    system2d_gpu.createField("iqx2dgpu", false);
    system2d_gpu.createField("iqy2dgpu", false);
    system2d_gpu.createField("invq2dgpu", false);
    system2d_gpu.createField("lapinvq2dgpu", false);
    
    system1d_cpu.addEquation("lap1dcpu = -q^2 * init1dcpu");
    system1d_cpu.addEquation("iqx1dcpu = iqx * init1dcpu");
    system1d_cpu.addEquation("invq1dcpu = 1/q * init1dcpu");
    system1d_cpu.addEquation("lapinvq1dcpu = q^2 * 1/q * init1dcpu");

    system1d_gpu.addEquation("lap1dgpu = -q^2 * init1dgpu");
    system1d_gpu.addEquation("iqx1dgpu = iqx * init1dgpu");
    system1d_gpu.addEquation("invq1dgpu = 1/q * init1dgpu");
    system1d_gpu.addEquation("lapinvq1dgpu = q^2 * 1/q * init1dgpu");

    system2d_cpu.addEquation("lap2dcpu = -q^2 * init2dcpu");
    system2d_cpu.addEquation("iqx2dcpu = iqx * init2dcpu");
    system2d_cpu.addEquation("invq2dcpu = 1/q * init2dcpu");
    system2d_cpu.addEquation("lapinvq2dcpu = q^2 * 1/q * init2dcpu");

    system2d_gpu.addEquation("lap2dgpu = -q^2 * init2dgpu");
    system2d_gpu.addEquation("iqx2dgpu = iqx * init2dgpu");
    system2d_gpu.addEquation("invq2dgpu = 1/q * init2dgpu");
    system2d_gpu.addEquation("lapinvq2dgpu = q^2 * 1/q * init2dgpu");

    // Random initial state
    float sigma = ((float)NX)/8.0f;
    float s2p = std::sqrt(2.0f*PI);
    float s2 = std::sqrt(2.0f);
    for (int j = 0; j < NY; j++)
    {
        system1d_cpu.fields[0]->real_array[j].x = 1.0f/(sigma * s2p) * std::exp(-((j-NY/2)*(j-NY/2))/(2.0f * sigma*sigma)); 
        system1d_gpu.fields[0]->real_array[j].x = 1.0f/(sigma * s2p) * std::exp(-((j-NY/2)*(j-NY/2))/(2.0f * sigma*sigma)); 
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            system2d_cpu.fields[0]->real_array[index].x = 1.0f/(sigma * s2p) * std::exp(-((i - NX/2)*(i-NX/2) + (j-NY/2)*(j-NY/2))/(2.0f * sigma*sigma)); 
            system2d_gpu.fields[0]->real_array[index].x = 1.0f/(sigma * s2p) * std::exp(-((i - NX/2)*(i-NX/2) + (j-NY/2)*(j-NY/2))/(2.0f * sigma*sigma)); 
        }
    }

    system1d_cpu.prepareProblem();
    system2d_cpu.prepareProblem();
    system1d_gpu.prepareProblem();
    system2d_gpu.prepareProblem();

    system1d_cpu.advanceTime();
    system1d_cpu.advanceTime();
 
    system2d_cpu.advanceTime();
    system2d_cpu.advanceTime();

    system1d_gpu.advanceTime();
    system1d_gpu.advanceTime();

    system2d_gpu.advanceTime();
    system2d_gpu.advanceTime();

    return 0;
}
