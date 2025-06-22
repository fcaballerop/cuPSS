#include <iostream>
#include "../inc/cupss.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define NX 128
#define NY 128 
#define NSTEPS 10000

int main(int argc, char **argv)
{
    evolver system(RUN_GPU, NX, NY, 1.0f, 1.0f, 0.1f, 100);

    system.createField("phi", true);
    system.createField("iqxphi", false);
    system.createField("iqyphi", false);
    system.createField("sigxx", false);
    system.createField("sigxy", false);
    system.createField("vx", false);
    system.createField("vy", false);
    system.createField("w", false);

    system.addParameter("a", -1.0f);
    system.addParameter("b", 1.0f);
    system.addParameter("k", 4.0f);
    system.addParameter("eta", 1.0f);
    system.addParameter("friction", 0.0f);
    system.addParameter("ka", -4.0f);
    system.addParameter("D", 0.01f);

    system.addEquation("dt phi + ( a *q^2 + k*q^4)*phi= - b* q^2* phi^3 -vx*iqxphi - vy*iqyphi");
    system.addEquation("iqxphi = iqx*phi");
    system.addEquation("iqyphi = iqy*phi");
    system.addEquation("sigxx = - 0.5*ka *iqxphi * iqxphi + 0.5*ka*iqyphi*iqyphi");
    system.addEquation("sigxy = - ka *iqxphi * iqyphi");

    system.addEquation("vx * (friction + eta*q^2) = (iqx + iqx^3*1/q^2 - iqx*iqy^2*1/q^2) * sigxx + (iqy + iqx^2* iqy*1/q^2 + iqx^2*iqy*1/q^2) * sigxy");
    system.addEquation("vy * (friction + eta*q^2) = (iqx + iqx*iqy^2*1/q^2 + iqx*iqy^2*1/q^2) * sigxy + (-iqy - iqy^3*1/q^2 + iqx^2*iqy*1/q^2) * sigxx");
    system.addEquation("w = 0.5*iqx * vy - 0.5*iqy*vx ");


    // Random initial state
    std::srand(1324);
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            system.fieldsReal["phi"][index].x = 0.001f * (float)(rand()%200-100);
        }
    }

    system.prepareProblem();
    system.fieldsMap["phi"]->outputToFile = true;

    int steps = NSTEPS;
    int check = steps/100;
    if (check < 1) check = 1;

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
