#include <cmath>
#include <driver_types.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <string>
#include <cuda_runtime.h>
#include <filesystem>
#include "../inc/cupss.h"

void evolver::common_constructor()
{
    currentTime = 0.0f;
    currentTimeStep = 0;
    dtsqrt = std::sqrt(dt);

    if (sz == 1)
    {
        if (sy == 1)
        {
            blocks = 1;
            threads_per_block = sx;
        }
        else 
        {
            threads_per_block = dim3(32,32);
            int bx = (sx+31)/32;
            int by = (sy+31)/32;
            // blocks = dim3(sx/32,sy/32);
            blocks = dim3(bx,by);
        }
    }
    else 
    {
        threads_per_block = dim3(32,32,32);
        int bx = (sx+31)/32;
        int by = (sy+31)/32;
        int bz = (sz+31)/32;

        blocks = dim3(bx,by,bz);
    }

    _parser = new parser(this);
}

evolver::evolver(bool _with_cuda, int _sx, float _dx, float _dt, int _ses) : sx(_sx), sy(1), sz(1), dx(_dx), dy(1.0f), dz(1.0f), dt(_dt), writeEveryNSteps(_ses)
{
    std::srand(time(NULL));
    with_cuda = _with_cuda;
    
    common_constructor();
}

evolver::evolver(bool _with_cuda, int _sx, int _sy, float _dx, float _dy, float _dt, int _ses) : sx(_sx), sy(_sy), sz(1), dx(_dx), dy(_dy), dz(1.0f), dt(_dt), writeEveryNSteps(_ses)
{
    std::srand(time(NULL));
    with_cuda = _with_cuda;

    common_constructor();
}

evolver::evolver(bool _with_cuda, int _sx, int _sy, int _sz, float _dx, float _dy, float _dz, float _dt, int _ses) : sx(_sx), sy(_sy), sz(_sz), dx(_dx), dy(_dy), dz(_dz), dt(_dt), writeEveryNSteps(_ses)
{
    std::srand(time(NULL));
    with_cuda = _with_cuda;

    common_constructor();
}

int evolver::createFromFile(const std::string &file)
{
    _parser->createFromFile(file);
    return 0;
}

void evolver::prepareProblem()
{
    bool created_data_dir = false;
    bool create_dir_exception = false;
    try 
    {
        created_data_dir = std::filesystem::create_directory("data");
    }
    catch(std::exception &e)
    {
        create_dir_exception = true;
        std::cout << "ERROR CREATING DATA DIRECTORY, is there a file called 'data'?" << std::endl;
    }
    if ((not created_data_dir) && (!create_dir_exception))
    {
        //data dir already exists.
    }
    // copy host to device to account for initial conditions
    for (int i = 0; i < fields.size(); i++)
    {
        cudaMemcpy(fields[i]->real_array_d, fields[i]->real_array, sx*sy*sz*sizeof(float2), cudaMemcpyHostToDevice);
        cudaMemcpy(fields[i]->comp_array_d, fields[i]->comp_array, sx*sy*sz*sizeof(float2), cudaMemcpyHostToDevice);
        fields[i]->toComp();
    }
    // for each field prepare device and precalculate implicits
    for (int i = 0; i < fields.size(); i++)
    {
        fields[i]->prepareDevice();
        fields[i]->precalculateImplicit(dt);
        fields[i]->system_p = this;
    }
}

void evolver::setOutputField(std::string _name, int _output)
{
    for (int i = 0; i < fields.size(); i++)
    {
        if (fields[i]->name == _name)
        {
            if (_output)
                fields[i]->outputToFile = true;
            else
                fields[i]->outputToFile = false;
            return;
        }
    }
    std::cout << "setOutputField EROR: " << _name << " not found." << std::endl;
}

int evolver::addParameter(std::string _name, float value)
{
    _parser->insert_parameter(_name, value);
    return 0;
}

int evolver::addEquation(std::string equation)
{
    _parser->add_equation(equation);
    return 0;
}

int evolver::existsField(std::string _name)
{
    int foundIndex = -1;
    for (int i = 0; i < fields.size(); i++)
    {
        if (fields[i]->name == _name)
        {
            foundIndex = i;
            return foundIndex;
        }
    }
    return foundIndex;
}

int evolver::addNoise(std::string _name, std::string equation)
{
    if (existsField(_name) == -1)
    {
        std::cout << "Adding noise to non existing field! (" << _name << ")" << std::endl;
        return -1;
    }
    pres prefactor = _parser->add_noise(equation);
    fieldsMap[_name]->isNoisy = true;
    fieldsMap[_name]->noise_amplitude = prefactor;
    return 0;
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
    field *newField = new field(sx, sy, sz, dx, dy, dz);
    newField->name = name;
    newField->isCUDA = with_cuda;
    newField->dynamic = dynamic;
    newField->blocks = blocks;
    newField->threads_per_block = threads_per_block;
    fields.push_back(newField);

    fieldsMap[name] = fields[fields.size()-1];
    fieldsReal[name] = fields[fields.size()-1]->real_array;
    fieldsFourier[name] = fields[fields.size()-1]->comp_array;
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
    std::cout << "With cuda " << with_cuda << std::endl;
}

void evolver::writeOut()
{
    if (with_cuda)
    {
        copyAllDataToHost();
    }
    // check for NaNs (not checked if no output)
    float c1 = fields[0]->real_array[0].x;
    if (c1 != c1)
    {
        std::cout << "NaN detected, exiting!" << std::endl;
        std::exit(-1);
    }
    for (int f = 0; f < fields.size(); f++)
    {
        if (fields[f]->outputToFile)
        {
            FILE *fp;
            // char *fileName = new char[50];
            // sprintf(fileName, "data/%s.csv.%i",fields[k]->name.c_str(), currentTimeStep);
            std::string fileName = "data/" + fields[f]->name + ".csv." + std::to_string(currentTimeStep);
            fp = fopen(fileName.c_str(), "w+");
            fprintf(fp, "x, y, z, %s\n", fields[f]->name.c_str());
            for (int k = 0; k < sz; k++)
            {
                for (int j = 0; j < sy; j++)
                {
                    for (int i = 0; i < sx; i++)
                    {
                        int index = k * sx * sy + j * sx + i;
                        fprintf(fp, "%i, %i, %i, %f\n", i, j, k, fields[f]->real_array[index].x);
                    }
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
    if (sz == 1)
    {
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
    }
    else 
    {
        std::cout << "3-dimensional system of size " << sx << "x" << sy << "x" << sz << std::endl;
        std::cout << "Physical size " << (float)sx*dx << "x"
            << (float)sy*dy << "x" << (float)sz*dz << " with cells of size " 
            << dx << "x" << dy << "x" << dz << std::endl;
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
                if (pre > 0.0f) implicitLine += "+";
                implicitLine += std::to_string(pre);
                if (fields[i]->implicit[j].iqx != 0)
                    implicitLine += "(iqx)^(" + std::to_string(fields[i]->implicit[j].iqx) + ")";
                if (fields[i]->implicit[j].iqy != 0)
                    implicitLine += "(iqy)^(" + std::to_string(fields[i]->implicit[j].iqy) + ")";
                if (fields[i]->implicit[j].iqz != 0)
                    implicitLine += "(iqz)^(" + std::to_string(fields[i]->implicit[j].iqz) + ")";
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
                if (fields[i]->terms[j]->prefactors_h[p].iqz != 0)
                    line += "(iqz)^(" + std::to_string(fields[i]->terms[j]->prefactors_h[p].iqz) + ")";
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
            std::cout << "+ sqrt[2*" << fields[i]->noise_amplitude.preFactor;
            if (fields[i]->noise_amplitude.q2n != 0)
                std::cout << "*q^" << fields[i]->noise_amplitude.q2n * 2;
            if (fields[i]->noise_amplitude.invq != 0)
                std::cout << "*1/|q|^" << fields[i]->noise_amplitude.q2n;
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

    term *newTerm = new term(sx, sy, sz, dx, dy, dz);
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

    newTerm->blocks = blocks;
    newTerm->threads_per_block = threads_per_block;

    fields[field_index]->terms.push_back(newTerm);
    return 0;
}

void evolver::copyAllDataToHost()
{
    for (int i = 0; i < fields.size(); i++)
    {
        cudaMemcpy(fields[i]->real_array, fields[i]->real_array_d, sx*sy*sz*sizeof(float2), cudaMemcpyDeviceToHost);
        // cudaMemcpy(fields[i]->comp_array, fields[i]->comp_array_d, sx*sy*sizeof(float2), cudaMemcpyDeviceToHost);
    }
}
