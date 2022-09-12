cmake_minimum_required(VERSION 3.10)

project(PoseLib VERSION 2.0.0)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Eigen
find_package(Eigen3 3.3.90 REQUIRED NO_MODULE)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# Benchmark
option(WITH_BENCHMARK "Build benchmark example." OFF)
#if(WITH_BENCHMARK)
	add_subdirectory(benchmark)
#endif()

# python bindings
option(PYTHON_PACKAGE "Build python package." ON)
#if(PYTHON_PACKAGE)
	add_subdirectory(pybind)
#endif()

# Compilation options
if(MSVC)
	target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${LIBRARY_NAME} PRIVATE -Wall -Werror -fPIC)
endif()
