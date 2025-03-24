#ifndef FIELD_H
#define FIELD_H

#include <random>
#include <vector>
#include <fftw3.h>
#include <string>

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>

#include "defines.h"

class term;

struct pres;

class evolver;

class field {
    private:
    const int sx, sy, sz;
    const float dx, dy, dz;
    const float stepqx, stepqy, stepqz;

    fftwf_plan plan_forward;
    fftwf_plan plan_backward;
    fftwf_plan plan_forward_dealias;
    fftwf_plan plan_backward_dealias;
    fftwf_plan noise_plan;
    cufftHandle plan_gpu;
    curandGenerator_t gen_d;
    cudaStream_t stream_d;
    curandRngType_t rng_d;
    curandOrdering_t order_d;
    // for dynamic change of parameters
    std::vector<std::string> implicit_prefactor_strings;
    
    public:
    field(int, float);
    field(int, int, float, float);
    field(int, int, int, float, float, float);
    ~field();
    void common_constructor();

    std::string name;
    std::map<std::string, int> usedParameters; // only implicits
    bool isCUDA;
    bool outputToFile;
    evolver *system_p;
    bool dynamic;
    int integrator;

    // Noise objects
    bool isNoisy;
    NoiseType noiseType;
    std::random_device rd;
    std::mt19937 rng;
    std::normal_distribution<> dist;

    // Naming convention for fields:
    // real_* : stores values in real space
    // comp_* : stores values in Fourier space
    // *_d    : device pointers (to GPU memory)
    // noise is split into _r and _i (real and imaginary)
    float2 *real_array;
    float2 *comp_array;
    float2 *real_array_d;
    float2 *comp_array_d;
    float2 **terms_h;
    float2 **terms_d;

    bool needsaliasing;
    int aliasing_order;
    float2 *comp_dealiased;
    float2 *real_dealiased;
    float2 *comp_dealiased_d;
    float2 *real_dealiased_d;

    float *gen_noise;
    float2 *noise_comp;
    float2 *noise_gend;
    float2 *noise_real;
    float2 *noise_fourier;
    float *noise_comp_d_r;
    float *noise_comp_d_i;


    std::vector<term *> terms;
    std::vector<pres> implicit;

    float *precomp_implicit;
    float *precomp_implicit_d;

    pres noise_amplitude;
    float *precomp_noise;
    float *precomp_noise_d;

    pres *implicit_terms;

    // callback functions 
    bool hasCB;
    void (*callback) (evolver *, float2 *, int, int, int);
    bool hasCBFourier;
    void (*callbackFourier) (evolver *, float2 *, int, int, int);

    dim3 threads_per_block;
    dim3 blocks;

    int setRHS(float);
    int updateTerms();
    void createNoise();
    void setToZero();
    void setNotDynamic(float);
    void setDynamic(float);

    void stepEuler(float);
    void stepRK2(float);
    void stepRK4(float);

    void toReal();
    void toComp();
    void normalize();
    void dealias();

    void copyHostToDevice();
    void copyDeviceToHost();
    void copyRealHostToDevice();
    void copyRealDeviceToHost();

    void writeToFile(int , int , int );

    void prepareDevice();

    void precalculateImplicit(float dt);

    float getStepqx();
    float getStepqy();
    float getStepqz();

    int addImplicitString(const std::string &);
    void printImplicitString();
    int updateParameter(const std::string &, float);
};

#endif // FIELD_H