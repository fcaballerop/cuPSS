cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(cuPSS LANGUAGES CXX CUDA)


add_library(cupss STATIC
    src/evolver.cu
    src/field.cu
    src/field_init.cu
    src/field_kernels.cu
    src/parser.cu
    src/term.cu
    src/term_init.cu
    src/term_kernels.cu
    src/boundary.cu
    src/boundary_kernels.cu
    )
set_property(TARGET cupss PROPERTY CUDA_ARCHITECTURES OFF)

INSTALL(TARGETS cupss
    LIBRARY DESTINATION /usr/lib/
    )

INSTALL(DIRECTORY
    "inc/"
    DESTINATION "/usr/include"
    )

