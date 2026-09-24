cmake_minimum_required(VERSION 3.18)
project(proxima LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -O3 without -ffast-math, deliberately. fast-math licenses the compiler to
# reassociate float math, which made l2sq() and l2sq_x4() round differently
# for the SAME pair (~5% of pairs, few-ulp deltas) despite identical written
# accumulation order. Mixed single/batched evaluations then made greedy
# descent comparisons inconsistent and it cycled forever on GIST1M's
# near-duplicate vectors. The hot kernels are hand-written NEON, so
# fast-math bought nothing worth that.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops -fno-omit-frame-pointer")

# Apple Silicon already enables NEON; nothing to add. We detect via __ARM_NEON in code.
# On x86 you'd want: -mavx2 -mfma (or -march=native).

# We use std::thread for parallelism (not OpenMP) to avoid the well-known
# macOS dyld error when two libomp copies are loaded, faiss-cpu bundles its
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
  src/ivfpq_fs.cpp
)

add_library(proxima_core STATIC ${CORE_SRC})
target_include_directories(proxima_core PUBLIC include)
target_compile_options(proxima_core PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(proxima_core PUBLIC Threads::Threads)

# Compile the scalar fallback paths even on ARM (CI runs both flavors; the
# fallbacks otherwise never get built on Apple Silicon dev machines).
option(PROXIMA_FORCE_SCALAR "Disable NEON paths, build scalar fallbacks" OFF)
if(PROXIMA_FORCE_SCALAR)
  target_compile_definitions(proxima_core PUBLIC PROXIMA_FORCE_SCALAR)
endif()

# AddressSanitizer smoke test: a standalone C++ harness that drives every
# index through add / remove_ids / re-add / search / save-load on both
# metrics with adversarial sizes. Builds the core + harness with ASan so CI
# catches the OOB/use-after-free class on every push (the v5 segfault would
# have tripped this). In this mode the pybind module is skipped, the job
# only runs the harness.
option(PROXIMA_ASAN "Build the ASan smoke-test executable instead of the module" OFF)
if(PROXIMA_ASAN)
  target_compile_options(proxima_core PUBLIC -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(proxima_core PUBLIC -fsanitize=address)
  add_executable(asan_smoke tests/cpp/asan_smoke.cpp)
  target_link_libraries(asan_smoke PRIVATE proxima_core)
else()
  pybind11_add_module(_proxima src/bindings.cpp)
  target_link_libraries(_proxima PRIVATE proxima_core)
  target_include_directories(_proxima PRIVATE include)
  if(PROXIMA_FORCE_SCALAR)
    target_compile_definitions(_proxima PRIVATE PROXIMA_FORCE_SCALAR)
  endif()
  # Drop the compiled module next to the Python package for in-place use.
  set_target_properties(_proxima PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/proxima
  )
endif()
