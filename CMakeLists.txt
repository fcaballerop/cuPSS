cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

project(cuPSS LANGUAGES CXX CUDA)

add_library(cupss STATIC
    src/evolver.cpp
    src/initializers.cpp
    src/field.cpp
    src/field_init.cpp
    src/field_kernels.cu
    src/parser.cpp
    src/term.cpp
    src/term_init.cpp
    src/term_kernels.cu
    src/cu_utils.cpp
    )
set_source_files_properties(src/evolver.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/initializers.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/field.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/field_init.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/parser.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/term.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/term_init.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(src/cu_utils.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET cupss PROPERTY CUDA_ARCHITECTURES OFF)

INSTALL(TARGETS cupss
    LIBRARY DESTINATION /usr/lib/
    )

INSTALL(DIRECTORY
    "inc/"
    DESTINATION "/usr/include"
    )

