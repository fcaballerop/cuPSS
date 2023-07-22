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

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define Nx 256
#define Ny 256

int main(int argc, char **argv)
{
    evolver system(1, Nx, Ny, 1.0f, 1.0f, 0.0001f, 100);

    system.createField("iqxphi", false);// 0
    system.createField("iqyphi", false);// 1
    system.createField("sigxx", false); // 2
    system.createField("sigxy", false); // 3
    system.createField("vx", false);    // 4
    system.createField("vy", false);    // 5
    system.createField("wxy", false);   // 6
    system.createField("phi", true);    // 7

    // CONSTANTS
    // v and Q
    float friction = 0.0f;
    float eta = 0.01f;
    // phi
    float a = -1.0f;
    float b = 1.0f;
    float phi0 = std::sqrt(-a/b);
    float k = 4.0f;
    float ka = k;
    float D = 0.01f;

    system.fields[7]->isNoisy = true;
    system.fields[7]->noiseType = GaussianWhite;
    system.fields[7]->noise_amplitude = {D, 1, 0, 0, 0};

    // Implicit terms
    system.fields[4]->implicit.push_back({eta, 1, 0, 0, 0});
    system.fields[5]->implicit.push_back({eta, 1, 0, 0, 0});
    system.fields[4]->implicit.push_back({friction, 0, 0, 0, 0});
    system.fields[5]->implicit.push_back({friction, 0, 0, 0, 0});
    system.fields[7]->implicit.push_back({-a, 1, 0, 0, 0});
    system.fields[7]->implicit.push_back({-k, 2, 0, 0, 0});

    //Explicit terms
    system.createTerm("iqxphi", {{1.0f, 0, 1, 0, 0}}, {"phi"});
    system.createTerm("iqyphi", {{1.0f, 0, 0, 1, 0}}, {"phi"});

    system.createTerm("sigxx", {{-ka/2.0f}}, {"iqxphi", "iqxphi"});
    system.createTerm("sigxx", {{ka/2.0f}}, {"iqyphi", "iqyphi"});
    system.createTerm("sigxy", {{-ka}}, {"iqxphi", "iqyphi"});

    system.createTerm("wxy", {{0.5f, 0, 1, 0, 0}}, {"vy"});
    system.createTerm("wxy", {{-0.5f, 0, 0, 1, 0}}, {"vx"});

    system.createTerm("phi", {{-b, 1, 0, 0, 0}}, {"phi", "phi", "phi"});
    system.createTerm("phi", {{-1.0f}}, {"vx", "iqxphi"});
    system.createTerm("phi", {{-1.0f}}, {"vy", "iqyphi"});

    // Terms for vx and vy
    pres iqx = {1.0f, 0, 1, 0, 0};
    pres iqy = {1.0f, 0, 0, 1, 0};
    pres miqy = {-1.0f, 0, 0, 1, 0};
    pres miqy3 = {-1.0f, 0, 0, 3, 2};
    pres iqx3 = {1.0f, 0, 3, 0, 2};
    pres iqx2iqy = {1.0f, 0, 2, 1, 2};
    pres miqxiqy2 = {-1.0f, 0, 1, 2, 2};
    pres iqxiqy2 = {1.0f, 0, 1, 2, 2};
    system.createTerm("vx", {iqx, iqx3, miqxiqy2}, {"sigxx"});
    system.createTerm("vx", {iqy, iqx2iqy, iqx2iqy}, {"sigxy"});
    system.createTerm("vy", {miqy, miqy3, iqx2iqy}, {"sigxx"});
    system.createTerm("vy", {iqx, iqxiqy2, iqxiqy2}, {"sigxy"});



    // Random initial state
    std::srand(time(NULL));
    float xi = std::sqrt(2.0f*k/a);
    float sxf = (float)Nx;
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            float ix = (float)i;
            float iy = (float)j;
            int index = j*Nx+i;
            if (j < Ny/2)
                system.fields[7]->real_array[index].x = -1.0f;
            else
                system.fields[7]->real_array[index].x = 1.0f;
            system.fields[7]->real_array[index].y = 0.0f;
        }
    }

    cudaMemcpy(system.fields[7]->real_array_d, system.fields[7]->real_array, Nx*Ny*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(system.fields[7]->comp_array_d, system.fields[7]->comp_array, Nx*Ny*sizeof(float2), cudaMemcpyHostToDevice);
    system.fields[7]->toComp();

    for (int i = 0; i < system.fields.size(); i++)
    {
        system.fields[i]->prepareDevice();
        system.fields[i]->precalculateImplicit(system.dt);
        system.fields[i]->outputToFile = false;
    }
    system.fields[4]->outputToFile = true;
    system.fields[5]->outputToFile = true;
    system.fields[6]->outputToFile = true;
    system.fields[7]->outputToFile = true;

    int steps = 1000000;
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
