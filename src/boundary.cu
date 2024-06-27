#include "boundary.h"


BoundaryConditions::BoundaryConditions(BoundaryType type, BoundaryDirection dimension, std::function<float(float,float,float)> value)
    :_type(type),_dimension(dimension),_value_fn(value),_single_value(false){}
     // 3d constructor
BoundaryConditions::BoundaryConditions(BoundaryType type, BoundaryDirection dimension, float value)
    :_type(type),_dimension(dimension),_value(value),_single_value(true){} //2d constructor

void BoundaryConditions::initalize(field * myField){
    // ok first thing we need to do is grab the pointer to the field
    // _field = myEvolver->fieldsMap[_fieldName]; // TO DO Name validation
    _fieldSize = myField->get_size();
    _fieldSpacing = myField->get_spaceing();
    _with_cuda = myField->isCUDA;
    _boundarySize = _fieldSize;
    _boundarySize[_dimension/2] = 1; // we don't need to iterate over the demension we're setting the boundary on


    // now that we have that we have dimensional information, we need to check to see if we need to calculate values for the boundary (if its not a single value)
    if (!_single_value) {
        // if we have to do that allocate the space here, figure out the values and then transfer them to the GPU if we're using it
        // the numebr of values we needs is the product of the dimensions that aren't this one
        
        long boundarySize = 1;
        for (long i = 0; i<3; i++){
           boundarySize*=_boundarySize[i];
        }
        _values = std::unique_ptr<float[]>(new float[_fieldSize[boundarySize]]);
        
        long index = 0;
        std::array<float,3> position;
        for (iz = 0; iz<_boundarySize[2]; iz++){
            for (iy = 0; iy<_boundarySize[1]; iy++){
                for (ix = 0; i<_boundarySize[0]; ix++){
                    position = {ix*_fieldSpacing[0],iy*_fieldSpacing[1],iz*_fieldSpacing[2]};
                    // if we're on the right boundary we need to correct the value of that position
                    if (dimension%2 == 1) {
                        // right boundary 
                        position[dimension/2]=(_fieldSize[dimension/2]-1)*_fieldSpacing[dimension/2];// hmmm might need to think carefully about how to encorpurate boundary layers here. for now ignore
                    }
                    _values[index] = _value_fn(position[0],position[1],position[2]);
                    index++;
                }
            }
        }
        
        if (_with_cuda){
            _threadDim=(32,32,32);
            switch (_dimension/2){
                case 0:
                    threadDim.x = 1;
                    break;
                case 1:
                    threadDim.y = 1;
                    break;
                case 2:
                    threadDim.z = 1;
                    break;
            }
            bx = (_boundarySize[0]+_threadDim.x-1)/_threadDim.x;
            by = (_boundarySize[1]+_threadDim.y-1)/_threadDim.y;
            bz = (_boundarySize[2]+_threadDim.z-1)/_threadDim.z;
            _blockDim = dim3(bx,by,bz);

            cudaMalloc(reinterpret_cast<void **>(&d_values), boundarySize * sizeof(float));
            cudaMemcpy(d_values, _values.get(), boundarySize * sizeof(float), cudaMemcpyHostToDevice);
        }
    }
}
long BoundaryConditions::flatten_index(std::array<int,3> dimension_index)
{
    // index = xi + nx *yi +nx *ny *zi 
    return dimension_index[0]+dimension_index[1]*_fieldSize[0]+dimension_index[2]*_fieldSize[0]*_fieldSize[1];
}
void BoundaryConditions::operator(float2* fieldValues)
{
    switch (_type){
        case BoundaryType::Dirichlet:
            applyDirichelt(fieldValues);
            break;
        case BoundaryType::VonNeumann:
            applyVonNeumann(fieldValues);
            break;

    }
} 
void BoundaryConditions::applyDirichelt(float2* fieldValues) 
{
    if (_with_cuda)
    {
        bool leftwall = !(_dimension%2)
        dim3 field_size = dim3(_fieldSize[0], _fieldSize[1], _fieldSize[2])
        dim3 boundary_size = dim3(_boundarySize[0], _boundarySize[1], _boundarySize[2])

        if (_single_value) 
        {
            applyDiricheltSingleValue_gpu(fieldValues,_value,_depth,_dimension/2, leftwall, field_size,  boundary_size,  _blockDim,  _threadDim);
        }
        else {
            applyDiricheltMultipleValue_gpu(fieldValues,_values,_depth,_dimension/2, leftwall, field_size,  boundary_size,  _blockDim,  _threadDim);
        }

    } 
    
    else 
    {
        long valueIndex = 0;
        long fieldIndex = 0;
        std::array<int,3> dimension_index;
        for (iz = 0; iz<_boundarySize[2]; iz++) {
            for (iy = 0; iy<_boundarySize[1]; iy++) {
                for (ix = 0; i<_boundarySize[0]; ix++) {
                    for (ib = 0; ib < _depth; ib ++) {
                        dimension_index = {ix,iy,iz};
                        if (dimension%2 == 0) {
                            // left wall
                            dimension_index[dimension/2]=ib;
                        }
                        if (dimension%1 == 1){
                            // right wall
                            dimension_index[dimension/2]=_fieldSize[dimension/2]-ib-1;
                        }

                        fieldIndex = flatten_index(dimension_index);
                        if (_single_value){// add aliased array
                            fieldValues[fieldIndex].x=_value;
                        } else {
                            fieldValues[fieldIndex].x=_values[valueIndex];
                        }

                    }
                    valueIndex++;
                }
            }
        }
    }
}
void BoundaryConditions::applyVonNeumann(float2* fieldValues){
    if (_with_cuda)
    {
        bool leftwall = !(_dimension%2)
        dim3 field_size = dim3(_fieldSize[0], _fieldSize[1], _fieldSize[2])
        dim3 boundary_size = dim3(_boundarySize[0], _boundarySize[1], _boundarySize[2])
        float h = _fieldSpacing[_dimension/2];
        if (_single_value) 
        {
            applyVonNuemannSingleValue_gpu(fieldValues,_value,_depth,_dimension/2, leftwall, field_size,  boundary_size,h,  _blockDim,  _threadDim);
        }
        else {
            applyVonNuemannMultipleValue_gpu(fieldValues,_values,_depth,_dimension/2, leftwall, field_size,  boundary_size, h ,  _blockDim,  _threadDim);
        }

    } else {
        long valueIndex = 0;
        long fieldIndex = 0;
        long fieldIndexOneIn = 0;
        std::array<int,3> dimension_index;
        std::array<int,3> dimension_index_one_in;
        for (iz = 0; iz<_boundarySize[2]; iz++) {
            for (iy = 0; iy<_boundarySize[1]; iy++) {
                for (ix = 0; i<_boundarySize[0]; ix++) {
                    for (ib = 0; ib < _depth; ib ++) {
                        dimension_index = {ix,iy,iz};
                        dimension_index_one_in=dimension_index;
                        // for von nueman we have to start from the inside and go out to keep the deriviate correct
                        if (dimension%2 == 0) {
                            // left wall
                            dimension_index[dimension/2]=(_depth-ib-1);
                            dimension_index_one_in[dimension/2]=(_depth-ib);

                        }
                        if (dimension%1 == 1){
                            // right wall
                            dimension_index[dimension/2]=_fieldSize[dimension/2]-1-(ib-_depth-1);
                            dimension_index_one_in[dimension/2]=fieldSize[dimension/2]-1-(ib-_depth);

                        }

                        fieldIndex = flatten_index(dimension_index);
                        fieldIndexOneIn = flatten_index(dimension_index_one_in);
                        // (x[one in]-x)/dx = value
                        // x = x[one in] - dx *value;
                        if (_single_value){
                            fieldValues[fieldIndex]=fieldValues[fieldIndexOneIn]-_fieldSpacing[dimension/2]*_value;
                        } else {
                            fieldValues[fieldIndex]=fieldValues[fieldIndexOneIn]-_fieldSpacing[dimension/2]*_values[valueIndex];
                        }

                    }
                    valueIndex++;
                }
            }
        }
    }
}