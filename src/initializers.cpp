#include <cmath>
#include <driver_types.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <ostream>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include "../inc/cupss.h"

void evolver::initializeUniform(std::string field, float value)
{
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found)
    {
       std::cout << "ERROR in initialize uniform, " << field << " not found" << std::endl;
       std::exit(1);
    }
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                fieldsMap[field]->real_array[index].x = value;
            }
        }
    }
}

void evolver::initializeUniformNoise(std::string field, float value)
{
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found)
    {
        std::cout << "ERROR in initialize uniform, " << field << " not found" << std::endl;
        std::exit(1);
    }
    srand(time(0));
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                fieldsMap[field]->real_array[index].x = value * 0.01f * (float)(rand()%200-100);
            }
        }
    }
}

void evolver::initializeNormalNoise(std::string field, float mean, float sigma)
{
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found)
    {
        std::cout << "ERROR in initialize uniform, " << field << " not found" << std::endl;
        std::exit(1);
    }
    srand(time(0));
    float v1 = 0.0;
    float v2 = 0.0;
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                // Use Box-Muller algorithm
                if (index%2 == 0)
                {
                    v1 = 0.01*(float)(rand()%100+1);
                    v2 = 0.01*(float)(rand()%100+1);
                    fieldsMap[field]->real_array[index].x = sigma*sigma*std::sqrt(-2.0*std::log(v1))*std::cos(2.0*PI*v2) + mean;
                    fieldsMap[field]->real_array[index+1].x = sigma*sigma*std::sqrt(-2.0*std::log(v1))*std::sin(2.0*PI*v2) + mean;
                }
            }
        }
    }
}

void evolver::initializeHalfSystem(std::string field, float val1, float val2, float xi, int direction)
{
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found)
    {
        std::cout << "ERROR in initialize half system, " << field << " not found" << std::endl;
        std::exit(1);
    }
    if (xi <= 0.0)
    {
        std::cout << "ERROR in initialize, interface width cannot be 0 or negative" << std::endl;
        std::exit(1);
    }
    if (direction < 1 || direction > 3)
    {
        std::cout << "ERROR in initialize, direction can be 1, 2 or 3 for x, y, z, respectively" << std::endl;
        std::exit(1);
    }
    srand(time(0));
    int rev_size = sx;
    if (direction == 2)
        rev_size = sy;
    if (direction == 3)
        rev_size = sz;
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                int ref = i;
                if (direction == 2)
                    ref = j;
                if (direction == 3)
                    ref = k;
                fieldsMap[field]->real_array[index].x = val1 + (val2-val1)*0.5*(1.0 + std::tanh((ref - rev_size/2)/(std::sqrt(2)*xi)));
            }
        }
    }
}

void evolver::initializeDroplet(std::string field, float val1, float val2, float radius, float xi, int p_x, int p_y, int p_z)
{
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found)
    {
        std::cout << "ERROR in initialize droplet, " << field << " not found" << std::endl;
        std::exit(1);
    }
    if (xi <= 0.0 || radius <= 0.0)
    {
        std::cout << "ERROR in initialize, droplet radius and interface width cannot be 0 or negative" << std::endl;
        std::exit(1);
    }
    int x_ = p_x % sx;
    int y_ = p_y % sy;
    int z_ = p_z % sz;
    srand(time(0));
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                float x_d = i - x_;
                float y_d = j - y_;
                float z_d = k - z_;
                float r_c = std::sqrt(x_d*x_d + y_d*y_d + z_d*z_d);
                fieldsMap[field]->real_array[index].x = val1 + (val2-val1)*0.5*(1.0 + std::tanh((r_c - radius)/(std::sqrt(2)*xi)));
            }
        }
    }
}

void evolver::addDroplet(std::string field, float val, float radius, float xi, int p_x, int p_y, int p_z)
{
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found)
    {
        std::cout << "ERROR in initialize droplet, " << field << " not found" << std::endl;
        std::exit(1);
    }
    if (xi <= 0.0 || radius <= 0.0)
    {
        std::cout << "ERROR in initialize, droplet radius and interface width cannot be 0 or negative" << std::endl;
        std::exit(1);
    }
    int x_ = p_x % sx;
    int y_ = p_y % sy;
    int z_ = p_z % sz;
    srand(time(0));
    for (int k = 0; k < sz; k++)
    {
        for (int j = 0; j < sy; j++)
        {
            for (int i = 0; i < sx; i++)
            {
                int index = k * sx * sy + j * sx + i;
                float x_d = i - x_;
                float y_d = j - y_;
                float z_d = k - z_;
                float r_c = std::sqrt(x_d*x_d + y_d*y_d + z_d*z_d);
                fieldsMap[field]->real_array[index].x += val*0.5*(1.0 + std::tanh((radius - r_c)/(std::sqrt(2)*xi)));
            }
        }
    }
}

void evolver::initializeFromFile(std::string field, std::string file, int skiprows, char delimiter) {
    bool found = false;
    for (int i = 0; i < fields.size(); i++)
        if (field == fields[i]->name)
            found = true;
    if (!found) {
        std::cout << "ERROR in initialize droplet, " << field << " not found" << std::endl;
        std::exit(1);
    }
    FILE *chk = fopen(file.c_str(), "r"); // ANSI C style only used to check if file exists.
    if (chk == NULL) {
        std::cout << "Initial file: " << file << " not found. Doing nothing" << std::endl;
        std::exit(1);
    }
    else
        fclose(chk);

    std::ifstream input;
    std::string f_name;
    std::string posx;
    std::string posy = "0";
    std::string posz = "0";
    std::string value;
    input.open(file.c_str());

    for (int i = 0; i < skiprows; i++) {
        getline(input, posx); // throw away
    }
    
    int countlines = 0;
    for (int i = 0; i < sx*sy*sz; i++) {
        countlines++;
        if (!getline(input, posx, delimiter)) {
            std::cout << "Incompatible number of lines on file " << file << ", only " << countlines << " found for a system size of " << sx*sy*sz << std::endl;
            std::exit(1);
        }
        if (dimension > 1)
            getline(input, posy, delimiter);
        if (dimension > 2)
            getline(input, posz, delimiter);
        getline(input, value);

        int px = std::stoi(posx);
        int py = std::stoi(posy);
        int pz = std::stoi(posz);
        int index = pz*sx*sy + py*sx + px;
        float val = std::stof(value);
        fieldsMap[field]->real_array[index].x = val;
    }

    
}
