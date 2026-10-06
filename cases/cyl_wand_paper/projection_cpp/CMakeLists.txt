# projbench: the paper's projection comparison in C++ (see 2026-10-06-plan.md). A project of its own: it reads
# pyOpenLPT's RefractCurved sources and writes nothing outside this folder's build/.
cmake_minimum_required(VERSION 3.20)
project(projbench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)              # as pyOpenLPT's CMakeLists.txt
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# pyOpenLPT, read only: this folder is cases/cyl_wand_paper/projection_cpp inside it
get_filename_component(OPENLPT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# the interpreter build.py runs with (the refract env), found before pybind11 as pyOpenLPT's CMakeLists.txt does
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Our search, compiled as pyOpenLPT compiles it (cmake/openLPT.cmake: a static library, CMake's default Release flags)
add_library(pb_refract_curved STATIC "${OPENLPT_ROOT}/src/srcMath/RefractCurved.cpp")
target_include_directories(pb_refract_curved PUBLIC "${OPENLPT_ROOT}/inc/libMath")

# The iterative methods, compiled the same way
add_library(pb_baselines STATIC src/baselines.cpp)
target_include_directories(pb_baselines PUBLIC src)

pybind11_add_module(projbench src/module.cpp)
target_link_libraries(projbench PRIVATE pb_refract_curved pb_baselines)
target_compile_definitions(projbench PRIVATE PB_RELEASE_FLAGS="${CMAKE_CXX_FLAGS_RELEASE}")
