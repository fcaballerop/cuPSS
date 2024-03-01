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
    evolver system(1, Nx, Ny, 1.0f, 1.0f, 0.01f, 100);

    system.createField("iqxQxx", false);// 0
    system.createField("iqyQxx", false);// 1
    system.createField("iqxQxy", false);// 2
    system.createField("iqyQxy", false);// 3
    system.createField("sigxx", false); // 4
    system.createField("sigxy", false); // 5
    system.createField("vx", false);    // 6
    system.createField("vy", false);    // 7
    system.createField("wxy", false);   // 8
    system.createField("Q2", false);    // 9
    system.createField("Qxx", true);    // 10
    system.createField("Qxy", true);    // 11

    // Constants
    float eta = 1.0f;
    float aQ = -1.0f;
    float bQ = 1.0f;
    float kQ = 4.0f; // K=4 is the one used in the paper
    float lambda = 0.7f;
    float friction = 0.0f;
    float gamma = 1.0f;
    float alpha = -0.5f;

    // Implicit terms
    system.fields[6]->implicit.push_back({eta, 1, 0, 0, 0});
    system.fields[7]->implicit.push_back({eta, 1, 0, 0, 0});
    system.fields[6]->implicit.push_back({friction, 0, 0, 0, 0});
    system.fields[7]->implicit.push_back({friction, 0, 0, 0, 0});
    system.fields[10]->implicit.push_back({-aQ});
    system.fields[10]->implicit.push_back({-kQ, 1, 0, 0, 0});
    system.fields[11]->implicit.push_back({-aQ});
    system.fields[11]->implicit.push_back({-kQ, 1, 0, 0, 0});

    //Explicit terms
    system.createTerm("iqxQxx", {{1.0f, 0, 1, 0, 0}}, {"Qxx"});
    system.createTerm("iqyQxx", {{1.0f, 0, 0, 1, 0}}, {"Qxx"});
    system.createTerm("iqxQxy", {{1.0f, 0, 1, 0, 0}}, {"Qxy"});
    system.createTerm("iqyQxy", {{1.0f, 0, 0, 1, 0}}, {"Qxy"});

    system.createTerm("sigxx", {{alpha}}, {"Qxx"});
    system.createTerm("sigxy", {{alpha}}, {"Qxy"});

    system.createTerm("Qxx", {{lambda, 0, 1, 0, 0}}, {"vx"});
    system.createTerm("Qxx", {{-2.0f}}, {"Qxy", "wxy"});
    system.createTerm("Qxx", {{-bQ}}, {"Q2", "Qxx"});
    system.createTerm("Qxx", {{-1.0f}}, {"vx", "iqxQxx"});
    system.createTerm("Qxx", {{-1.0f}}, {"vy", "iqyQxx"});

    system.createTerm("Qxy", {{lambda/2, 0, 1, 0, 0}}, {"vy"});
    system.createTerm("Qxy", {{lambda/2, 0, 0, 1, 0}}, {"vx"});
    system.createTerm("Qxy", {{2.0f}}, {"Qxx", "wxy"});
    system.createTerm("Qxy", {{-bQ}}, {"Q2", "Qxy"});
    system.createTerm("Qxy", {{-1.0f}}, {"vx", "iqxQxy"});
    system.createTerm("Qxy", {{-1.0f}}, {"vy", "iqyQxy"});

    system.createTerm("wxy", {{0.5f, 0, 1, 0, 0}}, {"vy"});
    system.createTerm("wxy", {{-0.5f, 0, 0, 1, 0}}, {"vx"});

    system.createTerm("Q2", {{1.0f}}, {"Qxx", "Qxx"});
    system.createTerm("Q2", {{1.0f}}, {"Qxy", "Qxy"});

    // Terms for fx and fy
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
    std::srand(1324);
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            system.fields[10]->real_array[i*Ny+j].x = 0.0001f * (float)(std::rand() % 200 - 100);
            system.fields[10]->real_array[i*Ny+j].y = 0.0f;
            system.fields[11]->real_array[i*Ny+j].x = 0.0001f * (float)(std::rand() % 200 - 100);
            system.fields[11]->real_array[i*Ny+j].y = 0.0f;
        }
    }

    cudaMemcpy(system.fields[10]->real_array_d, system.fields[10]->real_array, Nx*Ny*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(system.fields[10]->comp_array_d, system.fields[10]->comp_array, Nx*Ny*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(system.fields[11]->real_array_d, system.fields[11]->real_array, Nx*Ny*sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(system.fields[11]->comp_array_d, system.fields[11]->comp_array, Nx*Ny*sizeof(float2), cudaMemcpyHostToDevice);
    system.fields[10]->toComp();
    system.fields[11]->toComp();

    for (int i = 0; i < system.fields.size(); i++)
    {
        system.fields[i]->prepareDevice();
        system.fields[i]->precalculateImplicit(system.dt);
        system.fields[i]->outputToFile = false;
        system.fields[i]->isNoisy = false;
    }
    system.fields[0]->outputToFile = true;
    system.fields[1]->outputToFile = true;
    system.fields[2]->outputToFile = true;
    system.fields[3]->outputToFile = true;
    system.fields[6]->outputToFile = true;
    system.fields[7]->outputToFile = true;
    system.fields[8]->outputToFile = true;
    system.fields[9]->outputToFile = true;
    system.fields[10]->outputToFile = true;
    system.fields[11]->outputToFile = true;

    int steps = 500000;
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
