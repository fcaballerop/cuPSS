#include <cupss.h>
#include <iostream>

void even_BC(evolver *sys, float2 *field, int sx, int sy, int sz) {
    for (int i = sx/2; i < sx; i++) {
        field[i].x = field[sx - i].x;
    }
}

void odd_BC(evolver *sys, float2 *field, int sx, int sy, int sz) {
    for (int i = sx/2; i < sx; i++) {
        field[i].x = -field[sx - i].x;
    }
}

int main() {
    float L = 1.0;
    int N = 128;
    float dx = L / (float)N;
    float dt = 0.001 * dx;

    evolver sys_dirichlet(RUN_CPU, 2 * N, dx, dt, 100);

    sys_dirichlet.createField("phi_dir", true);
    sys_dirichlet.initializeDroplet("phi_dir", 1.0, 0.0, 20.0, 3.0, N / 2, 0, 0);

    sys_dirichlet.fieldsMap["phi_dir"]->hasCB = true;
    sys_dirichlet.fieldsMap["phi_dir"]->callback = odd_BC;

    sys_dirichlet.addEquation("dt phi_dir + q^2*phi_dir = 0.0");
    sys_dirichlet.setOutputField("phi_dir", true);

    sys_dirichlet.prepareProblem();

    evolver sys_neumann(RUN_CPU, 2 * N, dx, dt, 100);

    sys_neumann.createField("phi_neu", true);
    sys_neumann.initializeDroplet("phi_neu", 1.0, 0.0, 20.0, 3.0, N / 2, 0, 0);

    sys_neumann.fieldsMap["phi_neu"]->hasCB = true;
    sys_neumann.fieldsMap["phi_neu"]->callback = even_BC;

    sys_neumann.addEquation("dt phi_neu + q^2*phi_von = 0.0");
    sys_neumann.setOutputField("phi_neu", true);

    sys_neumann.prepareProblem();

    int nsteps = 6000;
    for (int t = 0; t < nsteps; t++) {
        sys_dirichlet.advanceTime();
        sys_neumann.advanceTime();
        if (t * 100 % nsteps == 0) {
            std::cout << "\r" << t * 100 / nsteps;
            std::cout.flush();
        }
    }
}
