/*      CAHN-HILLIARD in 2D
 * This solver implements the Cahn-Hilliard equation in a 2D lattice.
 * This equation describes the evolution of a field phi through the PDE:
 * 
 * dt phi = \nabla^2 (a*phi + b*phi^3 -k \nabla^2 phi)
 * 
 * or in Fourier space:
 *
 * dt phi = -q^2 * (a*phi + b*phi^3 + k*q^2*phi)
 *
 * where the phi^3 represents now a convolution in q-space.
 */

#include <iostream>
#include "../inc/cupss.h"
#include <functional>
#include <cuda.h>
#include <cuda_runtime.h>


int main(int argc, char **argv)
{
    int NX = 128, NY = 128;
    float dx = 1.0, dy = 1.0;
    float dt = 0.1;
    float output_every_n_steps = 1000;

    evolver system(1, NX, NY, dx, dy, dt, output_every_n_steps);

    system.createField("phi", true);
    std::function<double(double,double,double)> boundary_Value_x =[](double x, double y, double z){return y/128;};

    std::function<double(double,double,double)> boundary_Value_y =[](double x, double y, double z){return x/128;};
    system.addBoundaryCondition("phi",BoundaryConditions(BoundaryType::Dirichlet,BoundaryDirection::xminus,0));
    system.addBoundaryCondition("phi",BoundaryConditions(BoundaryType::Dirichlet,BoundaryDirection::xplus,boundary_Value_x));
    system.addBoundaryCondition("phi",BoundaryConditions(BoundaryType::VonNeumann,BoundaryDirection::yminus,0));
    system.addBoundaryCondition("phi",BoundaryConditions(BoundaryType::VonNeumann,BoundaryDirection::yplus,boundary_Value_y));
    

    system.addParameter("a", -1.0);
    system.addParameter("b",  1.0);
    system.addParameter("k",  4.0);

    system.addEquation("dt phi + q^2*(a + k*q^2)*phi= - b*q^2*phi^3");

    system.initializeUniformNoise("phi", 0.01);

    system.prepareProblem();

    system.setOutputField("phi", true);

    int steps = 10000;
    int check = steps/100 < 1 ? 1 : steps/100;

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
