#ifndef EVOLVER_H
#define EVOLVER_H

#include <vector>
#include <string>

#include "defines.h"

class field;
class parser;

struct pres;

class evolver
{
private:
    const int sx, sy, sz;
    const float dx, dy, dz;
    const int writeEveryNSteps;
    bool with_cuda;
    float currentTime;
    int currentTimeStep;
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
    ~evolver();
    void common_constructor();

    // Direct access to fields
    std::map<std::string, field *> fieldsMap;
    std::map<std::string, float2 *> fieldsReal;
    std::map<std::string, float2 *> fieldsFourier;

    // Getters
    int getSystemSizeX();
    int getSystemSizeY();
    int getSystemSizeZ();
    float getSystemPhysicalSizeX();
    float getSystemPhysicalSizeY();
    float getSystemPhysicalSizeZ();
    int getCurrentTimestep();
    float getCurrentTime();
    bool getCuda();
    float getParameter(const std::string &);
    void printInformation();

    // System declaration functions
    void addField(field *);
    int createField(std::string, bool);
    int createTerm(const std::string &, const std::vector<pres> &, const std::vector<std::string> &);
    int addParameter(const std::string &, float);
    int addEquation(const std::string &);
    int addNoise(const std::string &, const std::string &);
    int createFromFile(const std::string &);
    int existsField(const std::string &);

    // Dynamics functions
    int advanceTime();
    void writeOut();
    void copyAllDataToHost();
    void setOutputField(const std::string &, int);
    int updateParameter(const std::string &, float);
    int writePrecision;
    bool writeParametersOnUpdate;

    // Initializing functions
    void prepareProblem();

    void initializeUniform(std::string, float);
    void initializeUniformNoise(std::string, float);
    void initializeNormalNoise(std::string, float, float);
    void initializeHalfSystem(std::string, float, float, float, int);
    void initializeDroplet(std::string, float, float, float, float, int, int, int);
    void addDroplet(std::string, float, float, float, int, int, int);
    void initializeFromFile(std::string field, std::string file, int skiprows, char delimiter);
};

#endif // EVOLVER_H
