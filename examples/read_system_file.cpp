#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
#include "../inc/cupss.h"

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 256
#define NY 256 

int main(int argc, char **argv)
{
    evolver system(1, NX, NY, 1.0f, 1.0f, 0.1f, 100);

    system.createFromFile("examples/modelh.in");

    system.printInformation();

    // Random initial state
    std::srand(1324);
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            system.fields[0]->real_array[j * NX + i].x = 0.001f * (float)(rand()%200-100);

    system.prepareProblem();

    for (int i = 0; i < 10000; i++)
    {
        system.advanceTime();
        if (i % 100 == 0)
        {
            std::cout << "Progress: " << i/100 << "%\r";
            std::cout.flush();
        }
    }

    return 0;
}
