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
#include "../inc/cupss.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define NX 512
#define NY 512

int main(int argc, char **argv)
{
    evolver system(1, NX, NY, 20.0f/128.0f, 20.0f/128.0f, 0.0001f, 5000);

    system.createField("phi", true);        // 0

    // Terms for field phi
    system.createTerm("phi", {{-1.0f, 1, 0, 0, 0}}, {"phi", "phi", "phi"});

    system.fields[0]->implicit.push_back({1.0f, 1, 0, 0, 0});
    system.fields[0]->implicit.push_back({-1.0f, 2, 0, 0, 0});

    // Random initial state
    std::srand(1324);
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            system.fields[0]->real_array[index].x = -0.0f + 0.001f * (float)(rand()%200-100);
        }
    }

    system.prepareProblem();

    system.fields[0]->isNoisy = false;
    system.fields[0]->noiseType = GaussianWhite;
    system.fields[0]->noise_amplitude = {0.1f, 1, 0, 0, 0};
    
    for (int i = 0; i < system.fields.size(); i++)
    {
        system.fields[i]->prepareDevice();
        system.fields[i]->precalculateImplicit(system.dt);
    }
    system.fields[0]->outputToFile = true;

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
