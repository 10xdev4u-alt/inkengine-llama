cmake_minimum_required(VERSION 3.16)
project(odsl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The bit-exact LUT header is read from the sibling anchor repo; it is never
# modified. Override ODSL_INK_INCLUDE if the anchor lives elsewhere.
set(ODSL_INK_INCLUDE "/tmp/build/inkengine/include" CACHE PATH
    "Directory containing ink_lut.hpp from the inkengine anchor")

# Override with -DODSL_VECTORS=/path/to/vectors.bin if needed.
set(ODSL_VECTORS "/tmp/build/inkengine/tests/vectors.bin" CACHE FILEPATH
    "Bit-exact NVFP4 test vector file")

add_executable(odsl odsl.cpp)
target_include_directories(odsl PRIVATE "${ODSL_INK_INCLUDE}")
target_compile_definitions(odsl PRIVATE ODSL_VECTORS="${ODSL_VECTORS}")

enable_testing()
add_test(NAME odsl_numeric_layout_model COMMAND odsl)
