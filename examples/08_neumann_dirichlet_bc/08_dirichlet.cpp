#include <iostream>
#include <cupss.h>

int progressBar(int current, int total) {
    if ((current * 100) % total == 0) {
        std::cout << "\r" << (current * 100)/total << "%";
        std::cout << std::flush;
    }
    return total;
}

void odd_BC(evolver * sys, float2 * field, int sx, int sy, int sz) {
    for (int i = 1; i < 128; i++) {
        field[i + 128].x = -field[128 - i].x;
    }
    field[0].x = 0.0;
    field[128].x = 0.0;
}

int main() {
    float A = 1.0;
    float B = 2.0;
    float eta = 0.01;
    float L = 1.0;
    int N = 128;
    float dx = L/(float)N;
    float dt = 0.001*dx;
    
    evolver sys(RUN_CPU, 2*N, dx, dt, 1);
    
    sys.createField("h", false);
    sys.createField("u", false);
    sys.createField("v", true);
    sys.initializeUniform("v", 1.5);
    sys.initializeUniform("u", 1.5);
    
    for (int i = 0; i < N; i++) {
        sys.fieldsMap["h"]->real_array[i].x = A * (1.0 - ((float)i)/((float)N)) + B * ((float)i)/((float)N);
        sys.fieldsMap["v"]->real_array[i].x -= sys.fieldsMap["h"]->real_array[i].x;
    }
    for (int i = N; i < 2*N; i++) {
        sys.fieldsMap["h"]->real_array[i].x = A * (1.0 - ((float)(i-N))/((float)N)) + B * ((float)(i-N))/((float)N);
        sys.fieldsMap["v"]->real_array[i].x -= sys.fieldsMap["h"]->real_array[i].x;
    }
    
    sys.addEquation("u = v + h");
    
    sys.fieldsMap["v"]->hasCB = true;
    sys.fieldsMap["v"]->callback = odd_BC;
    
    sys.addEquation("dt v + q^2*v = 0.0");
    
    
    sys.setOutputField("u", true);
    sys.setOutputField("v", true);
    sys.setOutputField("h", true);
    
    sys.prepareProblem();
    
    for (int t = 0; t < progressBar(t, 6000); t++)
        sys.advanceTime();
    std::cout << std::endl;
    return 0;
}
