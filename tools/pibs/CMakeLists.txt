cmake_minimum_required(VERSION 3.16)
project(pibs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(pibs pibs.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(pibs PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(test_pibs_wiring test_pibs_wiring.cpp)
target_include_directories(test_pibs_wiring PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # assert() is the test oracle; Release -DNDEBUG would silently disable it.
  target_compile_options(test_pibs_wiring PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
endif()

enable_testing()
add_test(NAME pibs_model COMMAND pibs)
add_test(NAME pibs_wiring COMMAND test_pibs_wiring)
