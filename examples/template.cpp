#include <iostream>
#include "../inc/cupss.h"
#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
    int NX = 128;
    int NY = 128;
    int NZ = 128;
    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
    float dt = 0.1f;
    int steps = 10000;
    int outputfreq = 100;
    int gpu = 1;

    evolver system(gpu, NX, NY, NZ, dx, dy, dz, dt, outputfreq);

    system.addParameter("D", 1.0f);
    
    system.createField("phi", true);
    
    system.addEquation("dt phi = 0");

    system.initializeUniformNoise("phi", 0.01f);

    system.prepareProblem();

    system.printInformation();

    for (int i = 0; i < steps; i++)
    {
        system.advanceTime();
    }

    return 0;
}
