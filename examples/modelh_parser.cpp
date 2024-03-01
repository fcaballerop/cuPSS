/* Short explanation on the way to add fields and terms in this file
 *
 * evolver is a class that merely calls updates on all fields and terms
 * the arguments on its constructor are 
 *
 *      evolver system(x,           sx,             sy,             dx,       dy,       dt);
 *                     Use CUDA | x-system size | y-system size | delta_x | delta_y | delta_t
 *
 * To this evolver we can add fields:
 *
 *      system.createField( name, dynamic );
 *
 * name is a string and dynamic if a boolean that sets whether the field
 * is set in each step through a time derivative or through an equality.
 *
 * To each field we can add terms
 *      
 *      system.createTerm(  field_name, prefactor, {field_1, ..., field_n}  );
 *
 *  This term would be a term of "field_name", with that prefactor, that multiplies
 *  fields field_1 to field_n.
 */ 

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
#include "../inc/parser.h"

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 256
#define NY 256 

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

    system.addParameter("a", -1.0f);
    system.addParameter("b", 1.0f);
    system.addParameter("k", 4.0f);
    system.addParameter("eta", 1.0f);
    system.addParameter("friction", 0.0f);
    system.addParameter("ka", -4.0f);
    system.addParameter("D", 0.01f);


    system.fields[0]->isNoisy = true;
    system.fields[0]->noiseType = GaussianWhite;
    system.fields[0]->noise_amplitude = {0.01f, 1, 0, 0, 0};

    system.addEquation("dt phi + ( a *q^2 + k*q^4)*phi= - b* q^2* phi^3 -vx*iqxphi - vy*iqyphi");
    system.addEquation("iqxphi = iqx*phi");
    system.addEquation("iqyphi = iqy*phi");
    system.addEquation("sigxx = - 0.5*ka *iqxphi * iqxphi + 0.5*ka*iqyphi*iqyphi");
    system.addEquation("sigxy = - ka *iqxphi * iqyphi");

    system.addEquation("vx * (friction + eta*q^2) = (iqx + iqx^3*1/q^2 - iqx*iqy^2*1/q^2) * sigxx + (iqy + iqx^2* iqy*1/q^2 + iqx^2*iqy*1/q^2) * sigxy");
    system.addEquation("vy * (friction + eta*q^2) = (iqx + iqx*iqy^2*1/q^2 + iqx*iqy^2*1/q^2) * sigxy + (-iqy - iqy^3*1/q^2 + iqx^2*iqy*1/q^2) * sigxx");
    system.addEquation("w = 0.5*iqx * vy - 0.5*iqy*vx ");

    system.printInformation();

    // Random initial state
    std::srand(1324);
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            system.fields[0]->real_array[index].x = -0.0f + 0.001f * (float)(rand()%200-100);
            system.fields[0]->real_array[index].y = 0.0f;
        }
    }

    cudaMemcpy(system.fields[0]->real_array_d, system.fields[0]->real_array, NX*NY*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(system.fields[0]->comp_array_d, system.fields[0]->comp_array, NX*NY*sizeof(float2), cudaMemcpyHostToDevice);
    system.fields[0]->toComp();

    for (int i = 0; i < system.fields.size(); i++)
    {
        system.fields[i]->prepareDevice();
        system.fields[i]->precalculateImplicit(system.dt);
        system.fields[i]->outputToFile = false;
    }
    system.fields[0]->outputToFile = true;

    int steps = 100000;
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
