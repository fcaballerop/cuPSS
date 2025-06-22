#include <iostream>
#include "../inc/cupss.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define NX 256
#define NY 256 
#define NZ 256

int main(int argc, char **argv)
{
    evolver system(RUN_GPU, NX, NY, NZ, 1.0f, 1.0f, 1.0f, 0.01f, 100);

    system.createField("phi", true);

    system.addParameter("a", -1.0f);
    system.addParameter("b", 1.0f);
    system.addParameter("k", 4.0f);

    system.addEquation("dt phi + ( a *q^2 + k*q^4)*phi= - b* q^2* phi^3 ");

    system.initializeUniformNoise("phi", 0.01);

    system.prepareProblem();

    system.setOutputField("phi", true);

    int steps = 100000;
    int check = steps/100;
    if (check < 1) check = 1;

    system.printInformation();

    for (int i = 0; i < steps; i++)
    {
        system.advanceTime();
        if (i % check == 0)
        {
            std::cout << "Progress: " << i/check << "%\r";
            std::cout.flush();
        }
    }

    return 0;
}
