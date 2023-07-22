#include <cmath>
#include <driver_types.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <string>
#include <cuda_runtime.h>
#include "../inc/evolver.h"
#include "../inc/field.h"
#include "../inc/term.h"

evolver::evolver(bool _with_cuda, int _sx, int _sy, float _dx, float _dy, float _dt, int _ses) : sx(_sx), sy(_sy), dx(_dx), dy(_dy), dt(_dt), writeEveryNSteps(_ses)
{
    std::srand(time(NULL));
    #ifdef WITHCUDA
    with_cuda = _with_cuda;
    #else
    std::cout << "Compiled without CUDA, ignoring GPU settings and running on CPU" << std::endl;
    with_cuda = false;
    #endif

    currentTime = 0.0f;
    currentTimeStep = 0;
    // writeEveryNSteps = 100;
    // sx = _sx;
    // sy = _sy;
    // dx = _dx;
    // dy = _dy;
    // dt = _dt;
    dtsqrt = std::sqrt(_dt);
}

void evolver::addField(field *newField)
{
    fields.push_back(newField);
}

int evolver::createField(std::string name, bool dynamic)
{
    for (int i = 0; i < fields.size(); i++)
    {
        if (fields[i]->name == name)
        {
            std::cout << "Trying to create field with name that already exists" << std::endl;
            return 1;
        }
    }
    field *newField = new field(sx, sy, dx, dy);
    newField->name = name;
    newField->isCUDA = with_cuda;
    newField->dynamic = dynamic;
    fields.push_back(newField);
    return 0;
}


int evolver::advanceTime()
{
    if (currentTimeStep % writeEveryNSteps == 0)
    {
        // Maybe calculate observables and write them out
        writeOut();
    }
    // Loop over each field 
        // Calculate RHSs
    for (int i = 0; i < fields.size(); i++)
    {
        if (!fields[i]->dynamic)
            fields[i]->updateTerms();
    }
    for (int i = 0; i < fields.size(); i++)
    {
        if (!fields[i]->dynamic)
            fields[i]->setRHS(dt); 
    }
    for (int i = 0; i < fields.size(); i++)
    {
        if (fields[i]->dynamic)
            fields[i]->updateTerms();
    }
    for (int i = 0; i < fields.size(); i++)
    {
        if (fields[i]->dynamic)
            fields[i]->setRHS(dt); 
    }

    currentTime += dt;
    currentTimeStep += 1;
    return 0;
}

void evolver::test()
{
    #ifdef WITHCUDA
    std::cout << "With cuda " << with_cuda << std::endl;
    #endif
}

void evolver::writeOut()
{
    if (with_cuda)
    {
        copyAllDataToHost();
    }
    for (int k = 0; k < fields.size(); k++)
    {
        if (fields[k]->outputToFile)
        {
            FILE *fp;
            // char *fileName = new char[50];
            // sprintf(fileName, "data/%s.csv.%i",fields[k]->name.c_str(), currentTimeStep);
            std::string fileName = "data/" + fields[k]->name + ".csv." + std::to_string(currentTimeStep);
            fp = fopen(fileName.c_str(), "w+");
            fprintf(fp, "x, y, %s\n", fields[k]->name.c_str());
            for (int j = 0; j < sy; j++)
            {
                for (int i = 0; i < sx; i++)
                {
                    int index = j * sx + i;
                    fprintf(fp, "%i, %i, %f\n", i, j, fields[k]->real_array[index].x);
                }
            }
            fclose(fp);
        }
    }
}

void evolver::printInformation()
{
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << "Information on this evolver:" << std::endl;
    if (sy == 1)
    {
        std::cout << "1-dimensional system of size N = " << sx << "." << std::endl;
        std::cout << "Physical size L = " << (float)sx*dx
                  << " with cells of size dx = " << dx << std::endl;
    }
    else 
    {
        std::cout << "2-dimensional system of size " << sx << "x" << sy << std::endl;
        std::cout << "Physical size " << (float)sx*dx << "x"
                  << (float)sy*dy << " with cells of size " 
                  << dx << "x" << dy << std::endl;
    }
    std::cout << "There are " << fields.size() << " fields." << std::endl;
    for (int i = 0; i < fields.size(); i++)
    {
        std::cout << "Field " << i << ": " << fields[i]->name;
        if (fields[i]->dynamic) std::cout << " is dynamic.";
        else std::cout << " is not dynamic";
        std::cout << " and has " << fields[i]->terms.size() << " explicit terms";
        std::cout << " and " << fields[i]->implicit.size() << " implicit terms.";
        std::cout << " Runs on GPU: " << fields[i]->isCUDA;
        if (fields[i]->needsaliasing)
            std::cout << ". Will be dealiased for a nonlinearity of order " << fields[i]->aliasing_order;
        else
            std::cout << ". Will not be dealised.";
        std::cout << std::endl << "\t";
        if (fields[i]->dynamic) std::cout << "(d/dt)";
        std::cout << fields[i]->name;
        if (fields[i]->dynamic) std::cout << " = ";

        if (fields[i]->implicit.size() > 0)
        {
            std::string implicitLine = "[";
            for (int j = 0; j < fields[i]->implicit.size(); j++)
            {
                float pre = fields[i]->implicit[j].preFactor;
                implicitLine += std::to_string(pre);
                if (fields[i]->implicit[j].iqx != 0)
                    implicitLine += "(iqx)^(" + std::to_string(fields[i]->implicit[j].iqx) + ")";
                if (fields[i]->implicit[j].iqy != 0)
                    implicitLine += "(iqy)^(" + std::to_string(fields[i]->implicit[j].iqy) + ")";
                if (fields[i]->implicit[j].q2n != 0)
                    implicitLine += "(q^2)^(" + std::to_string(fields[i]->implicit[j].q2n) + ")";
                if (fields[i]->implicit[j].invq != 0)
                    implicitLine += "(1/|q|)^(" + std::to_string(fields[i]->implicit[j].invq) + ")";
            }
            implicitLine += "]";
            if (fields[i]->dynamic) implicitLine += fields[i]->name;
            std::cout << implicitLine;
        }
        if (!fields[i]->dynamic)
            std::cout << " = ";
        for (int j = 0; j < fields[i]->terms.size(); j++)
        {
            std::string line = "";
            if(j != 0) line = " + [";
            else line = " [";
            for (int p = 0; p < fields[i]->terms[j]->prefactors_h.size(); p++)
            {
                float pre = fields[i]->terms[j]->prefactors_h[p].preFactor;
                line += " + (" + std::to_string(pre) + ")";
                if (fields[i]->terms[j]->prefactors_h[p].iqx != 0)
                    line += "(iqx)^(" + std::to_string(fields[i]->terms[j]->prefactors_h[p].iqx) + ")";
                if (fields[i]->terms[j]->prefactors_h[p].iqy != 0)
                    line += "(iqy)^(" + std::to_string(fields[i]->terms[j]->prefactors_h[p].iqy) + ")";
                if (fields[i]->terms[j]->prefactors_h[p].q2n != 0)
                    line += "(q^2)^(" + std::to_string(fields[i]->terms[j]->prefactors_h[p].q2n) + ")";
                if (fields[i]->terms[j]->prefactors_h[p].invq != 0)
                    line += "(1/|q|)^(" + std::to_string(fields[i]->terms[j]->prefactors_h[p].invq) + ")";
                if (p != fields[i]->terms[j]->prefactors_h.size()-1)
                    line += " + ";
            }
            line += "] ";
            line += "(";
            for (int k = 0; k < fields[i]->terms[j]->product.size(); k++)
                line += " " + fields[i]->terms[j]->product[k]->name;
            line += " )";
            std::cout << line;
        }
        if (fields[i]->isNoisy)
        {
            std::cout << "+ sqrt[2" << fields[i]->noise_amplitude.preFactor;
            // print amplitude function
            std::cout << "] x noise";
        }
        std::cout << std::endl << std::endl;
    }
}

// int evolver::createTerm(std::string _field, pres _prefactor, const std::vector<std::string> &_product)
// {
//     int field_index = -1;
//     for (int i = 0; i < fields.size(); i++)
//     {
//         if (fields[i]->name == _field)
//         {
//             field_index = i;
//             break;
//         }
//     }
//
//     if (field_index == -1)
//     {
//         std::cout << "Field " << _field << " not found trying to create term" << std::endl;
//         return 1;
//     }
//
//     term *newTerm = new term(sx, sy, dx, dy);
//     newTerm->isCUDA = with_cuda;
//
//     for (int i = 0; i < _product.size(); i++)
//     {
//         std::string fieldForProduct = _product[i];
//         for (int j = 0; j < fields.size(); j++)
//         {
//             if (fieldForProduct == fields[j]->name)
//             {
//                 newTerm->product.push_back(fields[j]);
//             }
//         }
//     }
//
//     newTerm->prefactors = _prefactor;
//
//     fields[field_index]->terms.push_back(newTerm);
//     return 0;
// }

int evolver::createTerm(std::string _field, const std::vector<pres> &_prefactors, const std::vector<std::string> &_product)
{
    int field_index = -1;

    for (int i = 0; i < fields.size(); i++)
    {
        if (fields[i]->name == _field)
        {
            field_index = i;
            break;
        }
    }

    if (field_index == -1)
    {
        std::cout << "Field " << _field << " not found trying to create term" << std::endl;
        return 1;
    }

    term *newTerm = new term(sx, sy, dx, dy);
    newTerm->isCUDA = with_cuda;

    for (int i = 0; i < _product.size(); i++)
    {
        std::string fieldForProduct = _product[i];
        for (int j = 0; j < fields.size(); j++)
        {
            if (fieldForProduct == fields[j]->name)
            {
                newTerm->product.push_back(fields[j]);
            }
        }
    }

    for (int i = 0; i < _prefactors.size(); i++)
    {
        newTerm->prefactors_h.push_back(_prefactors[i]);
    }

    fields[field_index]->terms.push_back(newTerm);
    return 0;
}

void evolver::copyAllDataToHost()
{
    for (int i = 0; i < fields.size(); i++)
    {
        cudaMemcpy(fields[i]->real_array, fields[i]->real_array_d, sx*sy*sizeof(float2), cudaMemcpyDeviceToHost);
        // cudaMemcpy(fields[i]->comp_array, fields[i]->comp_array_d, sx*sy*sizeof(float2), cudaMemcpyDeviceToHost);
    }
}
