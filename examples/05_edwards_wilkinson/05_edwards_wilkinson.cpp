#include "../../inc/cupss.h"
#include <iostream>

void progressBar(int current, int total);

int main() {
    int NX = 256;
    int NY = 256;
    float dx = 1.0;
    float dy = 1.0;
    float dt = 0.1;

    int steps = 10000;
    float write_frequency = 100;

    float D = 1.0;

    evolver system(RUN_GPU, NX, NY, dx, dy, dt, write_frequency);

    system.createField("phi", true);
    system.addParameter("D", D);
    system.addEquation("dt phi + D * q^2 * phi = 0");
    system.addNoise("phi", "2*D");

    system.setVerbose();
    system.setOutputField("phi", true);
    system.prepareProblem();

    for (int i = 0; i < steps; i++){
        progressBar(i, steps);
        system.advanceTime();
    }
    std::cout << std::endl;
    return 0;
}

void progressBar(int current, int total) {
    if ((current * 100) % total == 0) {
        std::cout << "\r" << (current * 100)/total << "%";
        std::cout << std::flush;
    }
}
