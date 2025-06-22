#include "../../inc/cupss.h"
#include <iostream>

void progressBar(int current, int total);

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "ERROR: system size must be provided\n";
        return 1;
    }
    int NX = std::stoi(argv[1]);
    float dx = 1.0;
    float dt = 0.01;

    int num_experiments = 100;
    int steps = 1000000;
    float write_frequency = 200;

    float D = 0.5;
    float l = 0.5;

    evolver system(RUN_CPU, NX, dx, dt, write_frequency);

    system.createField("h", true);
    system.createField("iqh", false);
    system.addParameter("D", D);
    system.addParameter("l", D);
    system.addEquation("dt h + 0.5 * q^2 * h = l * iqh^2");
    system.addEquation("iqh = iqx*h");
    system.addNoise("h", "2*D");

    system.initializeUniform("h", 0.0);
    system.prepareProblem();

    std::vector<float> width_vector(1000, 0.0);

    system.printInformation();

    for (int exp = 0; exp < num_experiments; exp++) {
        system.initializeUniform("h", 0.0);
        system.prepareProblem();
        progressBar(exp, num_experiments);
        for (int i = 0; i < steps; i++) {
            if ((i * 1000) % steps == 0) {
                // measure width
                float mean = system.fieldsMap["h"]->comp_array[0].x/(float)NX;
                // std::cout << mean << std::endl;
                float width = 0.0;
                for (int x = 0; x < NX; x++) {
                    float height = system.fieldsMap["h"]->real_array[x].x;
                    width += (height - mean)*(height-mean);
                }
                width /= (float)(2*NX);
                width_vector[(int)(((long)i*(long)1000/(long)steps))] += width;
            }
            system.advanceTime();
        }
    }
    for (int i = 0; i < width_vector.size(); i++) {
        width_vector[i] /= (float)num_experiments;
        std::cout << i * 1000 * dt << "\t" << width_vector[i] << std::endl;
    }
    return 0;
}

void progressBar(int current, int total) {
    if ((current * 100) % total == 0) {
        std::cout << "\r" << (current * 100)/total << "%";
        std::cout << std::flush;
    }
}
