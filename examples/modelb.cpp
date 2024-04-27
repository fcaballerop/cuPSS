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

#define NX 256
#define NY 256
#define NZ 256

int main(int argc, char **argv)
{
    std::cout << "Creating evolver\n";
    evolver system(1, NX, NY, NZ, 1.0f, 1.0f, 1.0f, 0.1f, 1000);

    std::cout << "Creating field\n";
    system.createField("phi", true);        // 0
                                            //
    // Terms for field phi
    std::cout << "Creating term\n";
    system.createTerm("phi", {{-1.0f, 1, 0, 0, 0, 0}}, {"phi", "phi", "phi"});

    std::cout << "Creating implicits\n";
    system.fields[0]->implicit.push_back({1.0f, 1, 0, 0, 0, 0});
    system.fields[0]->implicit.push_back({-4.0f, 2, 0, 0, 0, 0});

    // Random initial state
    std::srand(1324);
    std::cout << "Creating initial condition\n";
    for (int k = 0; k < NZ; k++)
    {
        for (int j = 0; j < NY; j++)
        {
            for (int i = 0; i < NX; i++)
            {
                int index = k * NX * NY + j * NX + i;
                system.fieldsReal["phi"][index].x = ((float)NX) * std::exp(-(((float)i - (float)NX/2.0f)*((float)i - (float)NX/2.0f) + ((float)j - (float)NY/2.0f)*((float)j - (float)NY/2.0f)+ ((float)k - (float)NZ/2.0f)*((float)k - (float)NZ/2.0f))/(0.01f * (float)(NX*NX*NX)));
                system.fields[0]->real_array[index].x = 0.0001f * (float)(rand()%200-100);
                // system.fieldsReal["phi"][index].x = 3.0f;
            }
        }
    }

    std::cout << "Preparing problem\n";
    system.prepareProblem();

    int steps = 10000;
    int check = steps/100;
    if (check < 1) check = 1;
    
    system.printInformation();

    std::cout << "Starting integration\n";
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
