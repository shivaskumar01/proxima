cmake_minimum_required(VERSION 3.18)
project(vectordb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive optimization. -ffast-math is OK here: distance kernels
# do simple FMAs over float32, and we don't depend on NaN/Inf semantics.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ffast-math -funroll-loops -fno-omit-frame-pointer")

# Apple Silicon already enables NEON; nothing to add. We detect via __ARM_NEON in code.
# On x86 you'd want: -mavx2 -mfma (or -march=native).

# We use std::thread for parallelism (not OpenMP) to avoid the well-known
# macOS dyld error when two libomp copies are loaded — faiss-cpu bundles its
# own and would conflict with Homebrew's. std::thread needs no extra deps.
find_package(Threads REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Discover pybind11 from the active Python env.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 CONFIG REQUIRED)

set(CORE_SRC
  src/distance.cpp
  src/flat.cpp
  src/kmeans.cpp
  src/hnsw.cpp
  src/ivfpq.cpp
)

add_library(vectordb_core STATIC ${CORE_SRC})
target_include_directories(vectordb_core PUBLIC include)
target_compile_options(vectordb_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(vectordb_core PUBLIC Threads::Threads)

pybind11_add_module(_vectordb src/bindings.cpp)
target_link_libraries(_vectordb PRIVATE vectordb_core)
target_include_directories(_vectordb PRIVATE include)

# Drop the compiled module next to the Python package for in-place use.
set_target_properties(_vectordb PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/vectordb
)
