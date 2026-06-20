# =============================================================================
# CMakeLists.txt — Master Input Layer
#   Unified input pipeline: Embedding (HFAQE) + Positional Encoding (HDPE)
# =============================================================================
# Targets
#   master_main — demo: end-to-end tokenize → embed → position-encode → train
#   master_test — integration validation suite
#
# Dependencies (sibling directories — provide via git submodules):
#   ../Component-1.2_Token-Embedding/src/  (HFAQE)
#   ../Component-1.3_Positional-Encoding/src/ (HDPE)
#
# Usage
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#   ./build/master_main
#   ./build/master_test
# =============================================================================

cmake_minimum_required(VERSION 3.16)
project(MasterInput VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Auto-init git submodules
# ---------------------------------------------------------------------------
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.gitmodules")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
        RESULT_VARIABLE GIT_SUBMOD_RESULT
    )
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(WARNING "git submodule update --init failed: ${GIT_SUBMOD_RESULT}")
    endif()
endif()

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ---------------------------------------------------------------------------
# Default build type
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
set(WARN_FLAGS
    -Wall -Wextra -Wpedantic
    -Wno-unused-variable
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -Wno-unused-function
)

set(OPT_FLAGS_RELEASE
    -O3 -funroll-loops -ffast-math -march=native
)

set(OPT_FLAGS_DEBUG
    -O0 -g3
)

# ---------------------------------------------------------------------------
# Compile definitions
# ---------------------------------------------------------------------------
set(DEFS "")

# ---------------------------------------------------------------------------
# Helper: configure target
# ---------------------------------------------------------------------------
function(master_configure_target target)
    target_compile_options(${target} PRIVATE
        ${WARN_FLAGS}
        $<$<CONFIG:Release>:${OPT_FLAGS_RELEASE}>
        $<$<CONFIG:Debug>:${OPT_FLAGS_DEBUG}>
    )
    target_compile_definitions(${target} PRIVATE ${DEFS})
    target_link_libraries(${target} PRIVATE m)
endfunction()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
set(SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Embedding and HDPE are included via relative paths in the source files.
# Add their src directories for any additional include needs.
set(HFAQE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Component-1.2_Token-Embedding/src)
set(HDPE_SRC_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/../Component-1.3_Positional-Encoding/src)

# ---------------------------------------------------------------------------
# Target 1: master_main
# Demo — end-to-end input pipeline.
# ---------------------------------------------------------------------------
add_executable(master_main ${SRC_DIR}/main.cpp)
master_configure_target(master_main)
target_include_directories(master_main PRIVATE
    ${SRC_DIR}
    ${HFAQE_SRC_DIR}
    ${HDPE_SRC_DIR}
)

# ---------------------------------------------------------------------------
# Target 2: master_test
# Integration validation suite.
# ---------------------------------------------------------------------------
add_executable(master_test ${TEST_DIR}/test.cpp)
master_configure_target(master_test)
target_include_directories(master_test PRIVATE
    ${SRC_DIR}
    ${HFAQE_SRC_DIR}
    ${HDPE_SRC_DIR}
)

# ---------------------------------------------------------------------------
# Target 3: train_input_layer
# Standalone trainer — runs the full input layer training loop.
# ---------------------------------------------------------------------------
add_executable(train_input_layer ${SRC_DIR}/train_input_layer.cpp)
master_configure_target(train_input_layer)
target_include_directories(train_input_layer PRIVATE
    ${SRC_DIR}
    ${HFAQE_SRC_DIR}
    ${HDPE_SRC_DIR}
)

# ---------------------------------------------------------------------------
# Target 4: test_train_numerical
# Numerical correctness validation (gradient, convergence, checkpoint).
# ---------------------------------------------------------------------------
add_executable(test_train_numerical ${TEST_DIR}/test_train_numerical.cpp)
master_configure_target(test_train_numerical)
target_include_directories(test_train_numerical PRIVATE
    ${SRC_DIR}
    ${HFAQE_SRC_DIR}
    ${HDPE_SRC_DIR}
)

# ---------------------------------------------------------------------------
# CTest integration
# ---------------------------------------------------------------------------
enable_testing()

add_test(
    NAME  MasterInput_Integration_Suite
    COMMAND master_test
)
set_tests_properties(MasterInput_Integration_Suite PROPERTIES
    TIMEOUT 600
    PASS_REGULAR_EXPRESSION "0 FAILED"
    FAIL_REGULAR_EXPRESSION "[1-9][0-9]* FAILED"
)

add_test(
    NAME  TrainNumerical_Suite
    COMMAND test_train_numerical
)
set_tests_properties(TrainNumerical_Suite PROPERTIES
    TIMEOUT 600
    PASS_REGULAR_EXPRESSION "0 FAILED"
    FAIL_REGULAR_EXPRESSION "[1-9][0-9]* FAILED"
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "----------------------------------------------------")
message(STATUS "Targets    : master_main  master_test")
message(STATUS "             train_input_layer  test_train_numerical")
message(STATUS "Run demo   : ./build/master_main")
message(STATUS "Run tests  : ./build/master_test")
message(STATUS "Train      : ./build/train_input_layer")
message(STATUS "Num checks: ./build/test_train_numerical")
message(STATUS "----------------------------------------------------")
