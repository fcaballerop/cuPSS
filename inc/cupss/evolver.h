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
    parser *_parser;
public:
    dim3 threads_per_block;
    dim3 blocks;
    float dt;
    float dtsqrt;
    std::vector<field*> fields;
    evolver(bool, int, float, float, int);
    evolver(bool, int, int, float, float, float, int);
    evolver(bool, int, int, int, float, float, float, float, int);
    void common_constructor();

    std::map<std::string, field *> fieldsMap;
    std::map<std::string, float2 *> fieldsReal;
    std::map<std::string, float2 *> fieldsFourier;

    void addField(field *);
    int createField(std::string, bool);
    // int createTerm(std::string, pres, const std::vector<std::string> &);
    int createTerm(std::string, const std::vector<pres> &, const std::vector<std::string> &);

    int addParameter(std::string, float);
    int addEquation(std::string);
    int addNoise(std::string, std::string);
    int createFromFile(const std::string &);

    int existsField(std::string);

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

    void setOutputField(std::string, int);

    void initializeUniform(std::string, float);
    void initializeUniformNoise(std::string, float);
};

#endif // EVOLVER_H
