cmake_minimum_required(VERSION 3.16)
project(roret_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(roret_probe roret_probe.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(roret_probe PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(test_union_cache test_union_cache.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # assert() is the test oracle; Release -DNDEBUG would silently disable it.
  target_compile_options(test_union_cache PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
endif()

add_executable(test_roret_traffic test_roret_traffic.cpp)
target_include_directories(test_roret_traffic PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../src")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_roret_traffic PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
endif()

add_executable(test_roret_tile_bitexact test_roret_tile_bitexact.cpp)
target_include_directories(test_roret_tile_bitexact PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(test_roret_tile_bitexact PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
endif()

add_executable(test_roret_union_census test_roret_union_census.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # assert() is the oracle here; Release -DNDEBUG would silently disable it.
  target_compile_options(test_roret_union_census PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
endif()

enable_testing()
add_test(NAME test_union_cache COMMAND test_union_cache)
add_test(NAME test_roret_traffic COMMAND test_roret_traffic)
add_test(NAME test_roret_tile_bitexact COMMAND test_roret_tile_bitexact)
add_test(NAME roret_union_census COMMAND test_roret_union_census)
