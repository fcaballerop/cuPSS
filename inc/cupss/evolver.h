#ifndef EVOLVER_H
#define EVOLVER_H

#include <vector>
#include <string>

#include "defines.h"

class field;
class parser;

struct pres;

class evolver {
private:
    const int sx, sy, sz;
    const float dx, dy, dz;
    const int writeEveryNSteps;
    bool with_cuda;
    float currentTime;
    int currentTimeStep;
    bool verbose;
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
    void setVerbose();
    void unsetVerbose();

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

    void initializeUniform(std::string field, float value);
    void initializeUniformNoise(std::string field, float amplitude);
    void initializeNormalNoise(std::string field, float mean, float sigma);
    void initializeHalfSystem(std::string field, float value1, float value2, float interface_width, int direction);
    void initializeDroplet(std::string field, float value_out, float value_in, float radius, float interface_width, int center_x, int center_y, int center_z);
    void addDroplet(std::string field, float value, float radius, float interface_width, int center_x, int center_y, int center_z);
    void initializeFromFile(std::string field, std::string file, int skiprows, char delimiter);
};

#endif // EVOLVER_H
