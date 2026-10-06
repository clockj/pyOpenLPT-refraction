# C++ unit tests of the stb_improve campaign, built by run_unit_tests.py in variants (storage size, index checks,
# AddressSanitizer) against a chosen pyOpenLPT source tree (OPENLPT_SRC: the working tree or HEAD's worktree).
cmake_minimum_required(VERSION 3.15)
project(stb_improve_cpp CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED OPENLPT_SRC)
  get_filename_component(OPENLPT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
endif()

add_executable(test_matrix_storage test_matrix_storage.cpp)
target_include_directories(test_matrix_storage PRIVATE
  "${OPENLPT_SRC}/inc" "${OPENLPT_SRC}/inc/libMath" "${OPENLPT_SRC}/src/srcMath")
