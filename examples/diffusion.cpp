/* This file implements a noisy diffusion solver, also known
 * as Edwards-Wilkinson model. For a field phi, the equation
 * of motion is
 *
 * dphi       / d^2phi   d^2phi \
 * ----  = D  | ------ + ------ |  + eta
 *  dt        \  dx^2     dy^2  /
 * 
 * where eta is a White Gaussian noise with variance 2D
 *
 */
#include <iostream>
#include "../inc/cupss.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define NX 256
#define NY 256 

int main(int argc, char **argv)
{
    evolver system(RUN_GPU, NX, NY, 1.0f, 1.0f, 0.1f, 100);

    system.createField("phi", true);        // 0
    system.createField("lphi", false);

    system.addParameter("D", 1.0f);

    system.addEquation("dt phi = lphi");
    system.addEquation("lphi = -D*q^2*phi");

    system.addNoise("phi", "0.0*D");

    // Initial state is a Gaussian distribution
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            system.fieldsReal["phi"][index].x = ((float)NX) * std::exp(-(((float)i - (float)NX/2.0f)*((float)i - (float)NX/2.0f) + ((float)j - (float)NY/2.0f)*((float)j - (float)NY/2.0f))/(0.01f * (float)(NX*NX)));
        }
    }

    system.prepareProblem();
    system.setOutputField("phi", true);

    int steps = 50000;
    int check = steps/100;
    if (check < 1) check = 1;
    
    system.printInformation();

    for (int i = 0; i < steps; i++)
    {
        system.advanceTime();
        if (i % check == 0)
        {
            // Simple progress bar
            std::cout << "Progress: " << i/check << "%\r";
            std::cout.flush();
        }
    }

    return 0;
}
