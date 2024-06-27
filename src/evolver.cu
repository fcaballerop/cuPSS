#include <cmath>
#include <driver_types.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <ostream>
#include <string>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include "../inc/cupss.h"
#include "evolver.h"

void evolver::common_constructor()
{
    currentTime = 0.0f;
    currentTimeStep = 0;
    dtsqrt = std::sqrt(dt);

    if (sz == 1)
    {
        if (sy == 1)
        {
            dimension = 1;
            blocks = 1;
            threads_per_block = sx;
        }
        else 
        {
            dimension = 2;
            threads_per_block = dim3(32,32);
            int bx = (sx+31)/32;
            int by = (sy+31)/32;
            // blocks = dim3(sx/32,sy/32);
            blocks = dim3(bx,by);
        }
    }
    else 
    {
        dimension = 3;
        threads_per_block = dim3(16, 8, 8);
        int bx = (sx+15)/16;
        int by = (sy+7)/8;
        int bz = (sz+7)/8;

        blocks = dim3(bx,by,bz);
    }

    writePrecision = 6;
    _parser = new parser(this);
}

evolver::evolver(bool _with_cuda, int _sx, float _dx, float _dt, int _ses) : sx(_sx), sy(1), sz(1), dx(_dx), dy(1.0f), dz(1.0f), dt(_dt), writeEveryNSteps(_ses)
{
    std::srand(time(0));
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
    struct stat info;
    std::string pathname = "data";

    if ( stat( pathname.c_str(), &info ) != 0 )
    {
        std::cout << "data directory not found, creating it.\n";
        int dir_err = mkdir(pathname.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (dir_err == -1) {
            std::cout << "Error creating data directory\n";
            std::exit(1);
        }
    }
    else if ( info.st_mode & S_IFDIR )
    {
        // data directory already exists, do nothing
    }
    else 
    {
        std::cout << "Can't create data directory, is there a file called data?\n";
        std::exit(1);
    }
    // copy host to device to account for initial conditions
    _parser->writeParamsToFile("data/parameter_list.txt");

    for (int i = 0; i < fields.size(); i++)
    {
        fields[i]->copyHostToDevice();
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

int evolver::addBoundaryCondition(std::string _name,BoundaryConditions BC)
{
    int fieldIndex = existsField(_name);
    if (fieldIndex != -1)
        fields[fieldIndex]->addBoundaryCondition(BC);
    
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

void evolver::writeOut()
{
    for (int f = 0; f < fields.size(); f++)
    {
        fields[f]->writeToFile(currentTimeStep, dimension, writePrecision);
    }
}

void evolver::printInformation()
{
    std::cout << std::fixed;
    std::cout << std::setprecision(3);
    std::cout << "Information on this evolver:" << std::endl;
    std::cout << dimension << "-dimensional system of size " << sx << "x" << sy << "x" << sz << std::endl;
    std::cout << "Physical size " << (float)sx*dx << "x"
        << (float)sy*dy << "x" << (float)sz*dz << " with cells of size " 
        << dx << "x" << dy << "x" << dz << std::endl;
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
                    implicitLine += "(q)^(" + std::to_string(2*fields[i]->implicit[j].q2n) + ")";
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
                    line += "(q)^(" + std::to_string(2*fields[i]->terms[j]->prefactors_h[p].q2n) + ")";
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
        fields[i]->copyDeviceToHost();
    }
}

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

int evolver::getSystemSizeX()
{
    return sx;
}
int evolver::getSystemSizeY()
{
    return sy;
}
int evolver::getSystemSizeZ()
{
    return sz;
}
float evolver::getSystemPhysicalSizeX()
{
    return ((float)sx)*dx;
}
float evolver::getSystemPhysicalSizeY()
{
    return ((float)sy)*dy;
}
float evolver::getSystemPhysicalSizeZ()
{
    return ((float)sz)*dz;
}
