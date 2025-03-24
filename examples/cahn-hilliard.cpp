/*      CAHN-HILLIARD in 2D
 * This solver implements the Cahn-Hilliard equation in a 2D lattice.
 * This equation describes the evolution of a field phi through the PDE:
 * dt phi = \nabla^2 (a*phi + b*phi^3 -k \nabla^2 phi)
 * or in Fourier space:
 * dt phi = -q^2 * (a*phi + b*phi^3 + k*q^2*phi)
 * where the phi^3 represents now a convolution in q-space.
 */

#include <iostream>
#include <cupss.h>

void printProgressbar(int step, int total);

int main(int argc, char **argv)
{
    int NX = 256, NY = 256;
    float dx = 1, dy = 1;
    float dt = 0.1;
    float output_every_n_steps = 100;

    evolver system(RUN_GPU, NX, NY, dx, dy, dt, output_every_n_steps);
    system.setVerbose();

    system.createField("phi", true);

    system.addParameter("a", -1.0);
    system.addParameter("b",  1.0);
    system.addParameter("k",  4.0);
    system.addParameter("D",  1.0);

    system.addEquation("dt phi + q^2*(a + k*q^2)*phi= - b*q^2*phi^3");
    system.addNoise("phi", "2*D*q^2");

    system.initializeNormalNoise("phi", 0.0, 0.1);
    system.setOutputField("phi", true);

    system.prepareProblem();
    int steps = 100000;
    for (int i = 0; i < steps; i++)
    {
        system.advanceTime();
        printProgressbar(i, steps);
    }

    return 0;
}

void printProgressbar(int step, int total) {
    int check = total/100 >= 1 ? total/100 : 1;
    if (step % check == 0) 
    {
        std::cout << "Progress: " << step/check << "%\r";
        std::cout.flush();
    }
}
