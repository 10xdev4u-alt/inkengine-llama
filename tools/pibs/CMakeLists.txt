cmake_minimum_required(VERSION 3.16)
project(pibs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(pibs pibs.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pibs PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME pibs_model COMMAND pibs)
