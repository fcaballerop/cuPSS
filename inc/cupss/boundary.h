#pragma once
// #include "evolver.h"
#include "field.h"
#include <string>
#include <functional>
#include <memory>
typedef enum boundaryDirection {xminus,xplus,yminus,yplus,zminus,zplus};
class evolver;
class BoundaryConditions {
// this class knows if its in the x,y or z direction
// this class has a function f(x,y,z) or f(x,y) that returns a value
// this class knows how thick the boundary layer is
// this class knows which feild its working on

// this will be subclassed into direcshlet and vonnueman
// those subclasses will then 
private:
    std::string _fieldName;
    field* _field=nullptr;
    boundaryDirection _dimension;
    bool _single_value;
    bool _with_cuda;
    std::function<float(float,float,float)> _value_fn;
    float _value;
    float std::unique_ptr<float[]> _values;
    float *d_values = nullptr; // for use when we have a BC that varies over space
    std::array<int,3> _iterateSize;
    std::array<int,3> _fieldSize;
    std::array<float,3> _fieldSpacing

    int _depth = 10; // the depth of the boundary layer
    long flatten_index(std::array<int,3>);

public:
    BoundaryConditions(boundaryDirection dimension, std::function<float(float,float,float)> value); // 3d constructor
    BoundaryConditions(boundaryDirection dimension, float value);
    void initalize(field*);
    virtual void operator() = 0; // applies the boundary condition
};
class Dirichlet : public BoundaryConditions
{
    void operator(float2*)
}
class VonNeumann: public BoundaryConditions
{
    void operator(float2*)
}