#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <ostream>
#include <chrono>
#include <ctime>
#include "../inc/cupss.h"

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 256
#define NY 256 
#define NSTEPS 10000

int main(int argc, char **argv)
{
    evolver system(1, NX, NY, 1.0f, 1.0f, 0.1f, 100);

    system.createField("phi", true);
    system.createField("iqxphi", false);
    system.createField("iqyphi", false);
    system.createField("sigxx", false);
    system.createField("sigxy", false);
    system.createField("vx", false);
    system.createField("vy", false);
    system.createField("w", false);
    system.createField("P", false);

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

    system.addEquation("-q^2*P = (iqx*iqx-iqy*iqy)*sigxx + 2.0 * iqx*iqy*sigxy");

    system.addEquation("vx * (friction + eta*q^2) = -iqx*P + iqx*sigxx + iqy*sigxy");
    system.addEquation("vy * (friction + eta*q^2) = -iqy*P + iqx*sigxy - iqy*sigxx");
    system.addEquation("w = 0.5*iqx * vy - 0.5*iqy*vx ");

    system.printInformation();

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

    system.setOutputField("phi", 1);

    int steps = NSTEPS;
    int check = steps/100;
    if (check < 1) check = 1;
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < steps; i++)
    {
        system.advanceTime();
        if (i % check == 0)
        {
            std::cout << "Progress: " << i/check << "%\r";
            std::cout.flush();
        }
    }

    auto end = std::chrono::system_clock::now();
 
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
 
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s"
              << std::endl;
    return 0;
}
