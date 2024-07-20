#include "../inc/cupss.h"
#include <iostream>

void check_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error" << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

void check_device()
{
    int nDevices;
    int runtimeVersion;
    int driverVersion;
    check_error(cudaGetDeviceCount(&nDevices));
    check_error(cudaRuntimeGetVersion(&runtimeVersion));
    check_error(cudaDriverGetVersion(&driverVersion));

    if (nDevices == 0)
    {
        std::cerr << "No CUDA devices found but trying to run on CUDA, exiting." << std::endl;
        std::exit(1);
    }

    cudaDeviceProp prop;
    std::cout << "Devices found:" << std::endl;
    for (int i = 0; i < nDevices; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
    }

    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." << (driverVersion%100)/10 << std::endl;
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion%100)/10 << std::endl;
    if (runtimeVersion > driverVersion)
    {
        std::cout << "WARNING: runtime version is not supported by driver. Solver might not work properly" << std::endl;
    }
}
