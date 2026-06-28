# cpp_spareribs

A `cuda` program to solve multiple copies of a stochastic differential equation simultaneously.

## Requirements

- A C++20-compatible compiler.
- [`cmake`](https://github.com/doctest/doctest) for building.
- The [`cuda` toolkit](https://developer.nvidia.com/cuda/toolkit) for the `cuda` api and `curand`.
- [`doctest`](https://github.com/doctest/doctest) for testing; it is automatically downloaded via `cmake`'s `FetchContent` if not found locally.

## Building

    cmake -Bbuild
    cmake --build build -j<simultaneous_jobs>

Optionally pass `-DBUILD_TESTING=OFF` to the first command to disable compilation of the tests.

## Running
    
Running an executable:

    build/main/<executable>

Running all tests:

    ctest --test-dir build
