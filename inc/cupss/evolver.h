#ifndef EVOLVER_H
#define EVOLVER_H

#include <vector>
#include <string>
#include <array>
#include "defines.h"
#include "boundary.h"

class field;
class parser;
struct pres;

class evolver
{
private:
    const int sx, sy, sz;
    const float dx, dy, dz;
    const int writeEveryNSteps;
public:
    int dimension;
    dim3 threads_per_block;
    dim3 blocks;
    float dt;
    float dtsqrt;
    std::vector<field*> fields;
    parser *_parser;
    evolver(bool, int, float, float, int);
    evolver(bool, int, int, float, float, float, int);
    evolver(bool, int, int, int, float, float, float, float, int);
    void common_constructor();
    std::map<std::string, field *> fieldsMap;
    std::map<std::string, float2 *> fieldsReal;
    std::map<std::string, float2 *> fieldsFourier;

    int getSystemSizeX();
    int getSystemSizeY();
    int getSystemSizeZ();
    float getSystemPhysicalSizeX();
    float getSystemPhysicalSizeY();
    float getSystemPhysicalSizeZ();

    void addField(field *);
    int createField(std::string, bool);
    // int createTerm(std::string, pres, const std::vector<std::string> &);
    int createTerm(const std::string &, const std::vector<pres> &, const std::vector<std::string> &);

    int addParameter(const std::string &, float);
    int addEquation(const std::string &);
    int addNoise(const std::string &, const std::string &);

    int addBoundaryCondition(const std::string &, BoundaryCondition);

    int addParameter(std::string, float);
    int addEquation(std::string);
    int addBoundaryCondition(std::string,BoundaryCondition);
    int addNoise(std::string, std::string);
  
    int createFromFile(const std::string &);
    

    float getParameter(const std::string &);

    int existsField(const std::string &);

    // Global variables
    bool with_cuda;

    float currentTime;
    int currentTimeStep;

    int advanceTime();

    void test();

    int writePrecision;
    void writeOut();

    void printInformation();

    void copyAllDataToHost();

    void prepareProblem();

    void setOutputField(const std::string &, int);

    int updateParameter(const std::string &, float);

    void initializeUniform(std::string, float);
    void initializeUniformNoise(std::string, float);
    void initializeNormalNoise(std::string, float, float);
    void initializeHalfSystem(std::string, float, float, float, int);
    void initializeDroplet(std::string, float, float, float, float, int, int, int);
    void addDroplet(std::string, float, float, float, int, int, int);
};

#endif // EVOLVER_H
