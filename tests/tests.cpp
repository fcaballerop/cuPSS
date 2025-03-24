#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "../inc/cupss.h"

const int Nx = 16;

bool loadValColumn(const std::string &filename, float* output, size_t expectedSize, size_t dim) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }
    
    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;  // Skip empty lines
        std::stringstream ss(line);
        std::string token;
        float col1, col2;
        
        // Read first column (num1)
        for (int i = 0; i < dim; i++) {
            if (std::getline(ss, token, ',')) {
                try {
                    col1 = std::stof(token);
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing num1 in line: " << line << std::endl;
                    continue;
                }
            } else {
                std::cerr << "Invalid format in line: " << line << std::endl;
                continue;
            }
        }
        
        // Read second column (num2)
        if (std::getline(ss, token, ',')) {
            try {
                col2 = std::stof(token);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing num2 in line: " << line << std::endl;
                continue;
            }
        } else {
            std::cerr << "Invalid format in line: " << line << std::endl;
            continue;
        }
        
        if (count < expectedSize) {
            output[count] = col2;
            count++;
        }
    }
    file.close();
    
    return (count == expectedSize);
}

TEST(cuPSS_Tests, OneDCPUInit) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_cpu_1d(RUN_CPU, 16, 1.0, 1.0, 1);
    system_cpu_1d.createField("phi_cpu_1d", true);
    system_cpu_1d.initializeDroplet("phi_cpu_1d", 0, 1, 16/8, 4, 16/2, 0, 0);
    system_cpu_1d.prepareProblem();

    float* fileData = new float[Nx];
    bool success = loadValColumn("base_truths/phi_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    
    // Compare each element using EXPECT_FLOAT_EQ (ideal for floating point comparisons).
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_cpu_1d.fieldsMap["phi_cpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    delete[] fileData;
}

TEST(cuPSS_Tests, TwoDCPUInit) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_cpu_2d(RUN_CPU, 16, 16, 1.0, 1.0, 1.0, 1);
    system_cpu_2d.createField("phi_cpu_2d", true);
    system_cpu_2d.initializeDroplet("phi_cpu_2d", 0, 1, 16/8, 4, 16/2, 16/2, 0);
    system_cpu_2d.prepareProblem();

    float* fileData = new float[Nx * Nx];
    
    // Load file data directly into fileData pointer.
    bool success = loadValColumn("base_truths/phi_2d", fileData, Nx * Nx, 2);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    
    // Compare each element using EXPECT_FLOAT_EQ (ideal for floating point comparisons).
    for (int i = 0; i < Nx * Nx; ++i) {
        EXPECT_NEAR(system_cpu_2d.fieldsMap["phi_cpu_2d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    delete[] fileData;
}

TEST(cuPSS_Tests, ThreeDCPUInit) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_cpu_3d(RUN_CPU, 16, 16, 16, 1.0, 1.0, 1.0, 1.0, 1);
    system_cpu_3d.createField("phi_cpu_3d", true);
    system_cpu_3d.initializeDroplet("phi_cpu_3d", 0, 1, 16/8, 4, 16/2, 16/2, 0);
    system_cpu_3d.prepareProblem();

    float* fileData = new float[Nx * Nx * Nx];
    
    // Load file data directly into fileData pointer.
    bool success = loadValColumn("base_truths/phi_3d", fileData, Nx * Nx * Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    
    // Compare each element using EXPECT_FLOAT_EQ (ideal for floating point comparisons).
    for (int i = 0; i < Nx * Nx * Nx; ++i) {
        EXPECT_NEAR(system_cpu_3d.fieldsMap["phi_cpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    delete[] fileData;
}

TEST(cuPSS_Tests, OneDGPUInit) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_cpu_1d(RUN_GPU, 16, 1.0, 1.0, 1);
    system_cpu_1d.createField("phi_cpu_1d", true);
    system_cpu_1d.initializeDroplet("phi_cpu_1d", 0, 1, 16/8, 4, 16/2, 0, 0);
    system_cpu_1d.prepareProblem();
    system_cpu_1d.copyAllDataToHost();

    float* fileData = new float[Nx];
    
    // Load file data directly into fileData pointer.
    bool success = loadValColumn("base_truths/phi_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    
    // Compare each element using EXPECT_FLOAT_EQ (ideal for floating point comparisons).
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_cpu_1d.fieldsMap["phi_cpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    delete[] fileData;
}

TEST(cuPSS_Tests, TwoDGPUInit) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_gpu_2d(RUN_GPU, 16, 16, 1.0, 1.0, 1.0, 1);
    system_gpu_2d.createField("phi_gpu_2d", true);
    system_gpu_2d.initializeDroplet("phi_gpu_2d", 0, 1, 16/8, 4, 16/2, 16/2, 0);
    system_gpu_2d.prepareProblem();
    system_gpu_2d.copyAllDataToHost();

    float* fileData = new float[Nx * Nx];
    
    // Load file data directly into fileData pointer.
    bool success = loadValColumn("base_truths/phi_2d", fileData, Nx * Nx, 2);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    
    // Compare each element using EXPECT_FLOAT_EQ (ideal for floating point comparisons).
    for (int i = 0; i < Nx * Nx; ++i) {
        EXPECT_NEAR(system_gpu_2d.fieldsMap["phi_gpu_2d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    delete[] fileData;
}

TEST(cuPSS_Tests, ThreeDGPUInit) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_gpu_3d(RUN_GPU, 16, 16, 16, 1.0, 1.0, 1.0, 1.0, 1);
    system_gpu_3d.createField("phi_gpu_3d", true);
    system_gpu_3d.initializeDroplet("phi_gpu_3d", 0, 1, 16/8, 4, 16/2, 16/2, 0);
    system_gpu_3d.prepareProblem();
    system_gpu_3d.copyAllDataToHost();

    float* fileData = new float[Nx * Nx * Nx];
    
    // Load file data directly into fileData pointer.
    bool success = loadValColumn("base_truths/phi_3d", fileData, Nx * Nx * Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    
    // Compare each element using EXPECT_FLOAT_EQ (ideal for floating point comparisons).
    for (int i = 0; i < Nx * Nx * Nx; ++i) {
        EXPECT_NEAR(system_gpu_3d.fieldsMap["phi_gpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    delete[] fileData;
}

TEST(cuPSS_Tests, OneDCPUOperators) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_cpu_1d(RUN_CPU, 16, 1.0, 1.0, 1);

    system_cpu_1d.createField("phi_cpu_1d", true);
    system_cpu_1d.createField("lapphi_cpu_1d", false);
    system_cpu_1d.createField("iqxphi_cpu_1d", false);
    system_cpu_1d.createField("invqphi_cpu_1d", false);

    system_cpu_1d.addEquation("dt phi_cpu_1d+q^2*phi_cpu_1d = iqxphi_cpu_1d^2");
    system_cpu_1d.addEquation("lapphi_cpu_1d = -q^2*phi_cpu_1d");
    system_cpu_1d.addEquation("iqxphi_cpu_1d =  iqx*phi_cpu_1d");
    system_cpu_1d.addEquation("invqphi_cpu_1d =  1/q*phi_cpu_1d");

    system_cpu_1d.initializeDroplet("phi_cpu_1d", 0, 1, 16/8, 4, 16/2, 0, 0);

    system_cpu_1d.prepareProblem();

    system_cpu_1d.advanceTime();

    float* fileData = new float[Nx];
    
    bool success = false;
    
    // iqx
    success = loadValColumn("base_truths/iqxphi_cpu_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_cpu_1d.fieldsMap["iqxphi_cpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // lap
    success = loadValColumn("base_truths/lapphi_cpu_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_cpu_1d.fieldsMap["lapphi_cpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // invq
    success = loadValColumn("base_truths/invqphi_cpu_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_cpu_1d.fieldsMap["invqphi_cpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    
    // Free allocated memory.
    delete[] fileData;
}

TEST(cuPSS_Tests, OneDGPUOperators) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_gpu_1d(RUN_GPU, 16, 1.0, 1.0, 1);

    system_gpu_1d.createField("phi_gpu_1d", true);
    system_gpu_1d.createField("lapphi_gpu_1d", false);
    system_gpu_1d.createField("iqxphi_gpu_1d", false);
    system_gpu_1d.createField("invqphi_gpu_1d", false);

    system_gpu_1d.addEquation("dt phi_gpu_1d+q^2*phi_gpu_1d = iqxphi_gpu_1d^2");
    system_gpu_1d.addEquation("lapphi_gpu_1d = -q^2*phi_gpu_1d");
    system_gpu_1d.addEquation("iqxphi_gpu_1d =  iqx*phi_gpu_1d");
    system_gpu_1d.addEquation("invqphi_gpu_1d =  1/q*phi_gpu_1d");

    system_gpu_1d.initializeDroplet("phi_gpu_1d", 0, 1, 16/8, 4, 16/2, 0, 0);

    system_gpu_1d.prepareProblem();

    system_gpu_1d.advanceTime();

    system_gpu_1d.copyAllDataToHost();

    float* fileData = new float[Nx];
    
    bool success = false;
    
    // iqx
    success = loadValColumn("base_truths/iqxphi_gpu_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_gpu_1d.fieldsMap["iqxphi_gpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // lap
    success = loadValColumn("base_truths/lapphi_gpu_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_gpu_1d.fieldsMap["lapphi_gpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // invq
    success = loadValColumn("base_truths/invqphi_gpu_1d", fileData, Nx, 1);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx; ++i) {
        EXPECT_NEAR(system_gpu_1d.fieldsMap["invqphi_gpu_1d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    
    // Free allocated memory.
    delete[] fileData;
}

TEST(cuPSS_Tests, ThreeDCPUOperators) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_cpu_3d(RUN_CPU, 16, 16, 16, 1.0, 1.0, 1.0, 1.0, 1);

    system_cpu_3d.createField("phi_cpu_3d", true);
    system_cpu_3d.createField("lapphi_cpu_3d", false);
    system_cpu_3d.createField("iqxphi_cpu_3d", false);
    system_cpu_3d.createField("iqyphi_cpu_3d", false);
    system_cpu_3d.createField("iqzphi_cpu_3d", false);
    system_cpu_3d.createField("invqphi_cpu_3d", false);

    system_cpu_3d.addEquation("dt phi_cpu_3d +q^2*phi_cpu_3d = iqxphi_cpu_3d^2");
    system_cpu_3d.addEquation("lapphi_cpu_3d = -q^2*phi_cpu_3d");
    system_cpu_3d.addEquation("iqxphi_cpu_3d =  iqx*phi_cpu_3d");
    system_cpu_3d.addEquation("iqyphi_cpu_3d =  iqy*phi_cpu_3d");
    system_cpu_3d.addEquation("iqzphi_cpu_3d =  iqz*phi_cpu_3d");
    system_cpu_3d.addEquation("invqphi_cpu_3d =  1/q*phi_cpu_3d");

    system_cpu_3d.initializeDroplet("phi_cpu_3d", 0, 1, 16/8, 4, 16/2, 16/2, 0);
    system_cpu_3d.prepareProblem();
    system_cpu_3d.advanceTime();

    float* fileData = new float[Nx*Nx*Nx];
    bool success = false;
    // iqx
    success = loadValColumn("base_truths/iqxphi_cpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_cpu_3d.fieldsMap["iqxphi_cpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // iqy
    success = loadValColumn("base_truths/iqxphi_cpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_cpu_3d.fieldsMap["iqxphi_cpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // iqz
    success = loadValColumn("base_truths/iqxphi_cpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_cpu_3d.fieldsMap["iqxphi_cpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // lap
    success = loadValColumn("base_truths/lapphi_cpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_cpu_3d.fieldsMap["lapphi_cpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // invq
    success = loadValColumn("base_truths/invqphi_cpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_cpu_3d.fieldsMap["invqphi_cpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    
    // Free allocated memory.
    delete[] fileData;
}

TEST(cuPSS_Tests, ThreeDGPUOperators) {
    // Allocate and initialize the in-memory data pointer.
    evolver system_gpu_3d(RUN_GPU, 16, 16, 16, 1.0, 1.0, 1.0, 1.0, 1);

    system_gpu_3d.createField("phi_gpu_3d", true);
    system_gpu_3d.createField("lapphi_gpu_3d", false);
    system_gpu_3d.createField("iqxphi_gpu_3d", false);
    system_gpu_3d.createField("iqyphi_gpu_3d", false);
    system_gpu_3d.createField("iqzphi_gpu_3d", false);
    system_gpu_3d.createField("invqphi_gpu_3d", false);

    system_gpu_3d.addEquation("dt phi_gpu_3d +q^2*phi_gpu_3d = iqxphi_gpu_3d^2");
    system_gpu_3d.addEquation("lapphi_gpu_3d = -q^2*phi_gpu_3d");
    system_gpu_3d.addEquation("iqxphi_gpu_3d =  iqx*phi_gpu_3d");
    system_gpu_3d.addEquation("iqyphi_gpu_3d =  iqy*phi_gpu_3d");
    system_gpu_3d.addEquation("iqzphi_gpu_3d =  iqz*phi_gpu_3d");
    system_gpu_3d.addEquation("invqphi_gpu_3d =  1/q*phi_gpu_3d");

    system_gpu_3d.initializeDroplet("phi_gpu_3d", 0, 1, 16/8, 4, 16/2, 16/2, 0);
    system_gpu_3d.prepareProblem();
    system_gpu_3d.advanceTime();
    system_gpu_3d.copyAllDataToHost();

    float* fileData = new float[Nx*Nx*Nx];
    
    bool success = false;
    
    // iqx
    success = loadValColumn("base_truths/iqxphi_gpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_gpu_3d.fieldsMap["iqxphi_gpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // iqy
    success = loadValColumn("base_truths/iqxphi_gpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_gpu_3d.fieldsMap["iqxphi_gpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // iqz
    success = loadValColumn("base_truths/iqxphi_gpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_gpu_3d.fieldsMap["iqxphi_gpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // lap
    success = loadValColumn("base_truths/lapphi_gpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_gpu_3d.fieldsMap["lapphi_gpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    // invq
    success = loadValColumn("base_truths/invqphi_gpu_3d", fileData, Nx*Nx*Nx, 3);
    ASSERT_TRUE(success) << "Failed to load the expected number of elements from file.";
    for (int i = 0; i < Nx*Nx*Nx; ++i) {
        EXPECT_NEAR(system_gpu_3d.fieldsMap["invqphi_gpu_3d"]->real_array[i].x, fileData[i], 1e-4)
            << "Mismatch at index " << i;
    }
    
    // Free allocated memory.
    delete[] fileData;
}

// Main function to run all tests.
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
