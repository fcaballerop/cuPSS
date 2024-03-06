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
    const int sx, sy;
    const float dx, dy;
    const int writeEveryNSteps;
    parser *_parser;
public:
    dim3 threads_per_block;
    dim3 blocks;
    float dt;
    float dtsqrt;
    std::vector<field*> fields;
    evolver(bool, int, int, float, float, float, int);

    void addField(field *);
    int createField(std::string, bool);
    // int createTerm(std::string, pres, const std::vector<std::string> &);
    int createTerm(std::string, const std::vector<pres> &, const std::vector<std::string> &);

    int addParameter(std::string, float);
    int addEquation(std::string);
    int createFromFile(const std::string &);

    // Global variables
    bool with_cuda;

    float currentTime;
    int currentTimeStep;

    int advanceTime();

    void test();

    void writeOut();

    void printInformation();

    void copyAllDataToHost();

    void prepareProblem();

    void setOutputField(std::string, int);
};

#endif // EVOLVER_H
