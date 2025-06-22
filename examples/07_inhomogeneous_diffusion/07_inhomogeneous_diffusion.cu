#include <cupss.h>
#include <iostream>

__global__ void even_BC_k(float2 *field, int sx, int sy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = j * sx + i;

    if (i < sx && j < sy) {
        int i_sym = sx - i;
        int j_sym = sy - j;
        if (i >= sx / 2 && j < sy / 2) {
            field[index].x = field[j * sx + i_sym].x;
        }
        if (i < sx / 2 && j >= sy / 2) {
          field[index].x = field[j_sym * sx + i].x;
        }
        if (i >= sx / 2 && j >= sy / 2) {
          field[index].x = field[j_sym * sx + i_sym].x;
        }
    }
}

void even_BC(evolver *sys, float2 *field, int sx, int sy, int sz) {
  even_BC_k<<<sys->blocks, sys->threads_per_block>>>(field, sx, sy);
}

__global__ void odd_BC_k(float2 *field, int sx, int sy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int index = j * sx + i;

    if (i < sx && j < sy) {
        int i_sym = sx - i;
        int j_sym = sy - j;
        if (i > sx / 2 && j < sy / 2) {
            field[index].x = -field[j * sx + i_sym].x;
        }
        if (i < sx / 2 && j > sy / 2) {
            field[index].x = -field[j_sym * sx + i].x;
        }
        if (i > sx / 2 && j > sy / 2) {
            field[index].x = field[j_sym * sx + i_sym].x;
        }
    }
}

void odd_BC(evolver *sys, float2 *field, int sx, int sy, int sz) {
    odd_BC_k<<<sys->blocks, sys->threads_per_block>>>(field, sx, sy);
}

int main() {
    float L = 1.0;
    int N = 128;
    float dx = L / (float)N;
    float dt = 0.001 * dx;

    evolver sys(RUN_GPU, 2 * N, 2 * N, dx, dx, dt, 100);

    sys.createField("v", true);
    sys.createField("iqxv", false);
    sys.initializeDroplet("v", 1.0, 0.0, 20.0, 3.0, N / 2, N / 2, 0);

    sys.createField("x", false);
    float delta = 0.1;
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= N; j++)
            sys.fieldsMap["x"]->real_array[j * 2 * N + i].x =
                1.0 * (1.0 + std::tanh((i - (1.0 - delta) * (N / 2 - 1)) /
                                   (delta * (N / 2 - 1))));
    // 1.0 + (float)(i) / (float)(N);

    sys.fieldsMap["v"]->hasCB = true;
    sys.fieldsMap["v"]->callback = even_BC;
    sys.fieldsMap["x"]->hasCB = true;
    sys.fieldsMap["x"]->callback = even_BC;

    sys.addEquation("dt v +0.5*q^2*v = iqx*x*iqxv + x*iqy^2*v");
    sys.addEquation("iqxv = iqx*v");

    sys.setOutputField("v", true);
    sys.setOutputField("x", true);

    sys.prepareProblem();

    int nsteps = 6000;
    for (int t = 0; t < nsteps; t++) {
        sys.advanceTime();
        if (t * 100 % nsteps == 0) {
            std::cout << "\r" << t * 100 / nsteps;
            std::cout.flush();
        }
    }
    return 0;
}
