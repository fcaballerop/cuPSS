cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

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
    )

# set_target_properties(cupss PROPERTIES PUBLIC_HEADER 
#     "inc/cupss.h;inc/cupss/defines.h;inc/cupss/evolver.h;inc/cupss/field.h;inc/cupss/term.h;inc/cupss/parser.h;inc/cupss/field_kernels.cuh;inc/cupss/term_kernels.cuh;"
#     )

INSTALL(TARGETS cupss
    LIBRARY DESTINATION /usr/lib/
    )

INSTALL(DIRECTORY
    "inc/"
    DESTINATION "/usr/include"
    )

# target_compile_features(cuPSS PUBLIC cxx_std_17)
