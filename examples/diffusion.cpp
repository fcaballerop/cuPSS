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

#define NX 256
#define NY 256 

int main(int argc, char **argv)
{
    evolver system(RUN_GPU, NX, NY, 1.0f, 1.0f, 0.1f, 100);

    system.createField("phi", true);

    system.addParameter("D", 1.0f);

    system.addEquation("dt phi +D*q^2*phi = 0");

    // Initial state (a tanh() bump)
    system.initializeDroplet("phi", NX, 0, NX/4, NX/16, NX/2, NY/2, 0);
    
    // Uncomment the next line to add thermal noise
    // system.addNoise("phi", "D");

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
