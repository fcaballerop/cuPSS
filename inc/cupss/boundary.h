#pragma once
// #include "evolver.h"
#include "field.h"
#include <string>
#include <functional>
#include <memory>
typedef enum {xminus,xplus,yminus,yplus,zminus,zplus} BoundaryDirection;
typedef enum {Dirichlet,VonNeumann} BoundaryType;

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
    BoundaryDirection _dimension;
    BoundaryType _type;
    bool _single_value = 1;
    bool _with_cuda;
    std::function<float(float,float,float)> _value_fn = [](float x, float y, float z){return 0;};
    float _value;
    float* _values =nullptr;
    float *d_values = nullptr; // for use when we have a BC that varies over space
    std::array<int,3> _boundarySize;
    std::array<int,3> _fieldSize;
    std::array<float,3> _fieldSpacing;

    dim3 _blockDim;
    dim3 _threadDim;

    int _depth = 10; // the depth of the boundary layer
    long flatten_index(std::array<int,3>);

public:
    BoundaryConditions(BoundaryType type, BoundaryDirection dimension, std::function<float(float,float,float)> value); // 3d constructor
    BoundaryConditions(BoundaryType type, BoundaryDirection dimension, float value);
    BoundaryConditions();
    void initalize(field*);
    void operator()(float2*); // applies the boundary condition
    void applyDirichlet(float2*);
    void applyVonNeumann(float2*);
};
