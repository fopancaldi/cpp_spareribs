cmake_minimum_required(VERSION 3.25)
project(
  cpp_spareribs
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

option(BUILD_TESTING "Build tests" ON)

include(CTest)
include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES native)

add_library(spareribs INTERFACE)
target_include_directories(spareribs
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(spareribs INTERFACE CUDA::curand)

add_subdirectory(main)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
