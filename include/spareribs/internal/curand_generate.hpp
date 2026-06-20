#pragma once

#include <cstddef>
#include <curand.h>

namespace spareribs {

inline curandStatus_t generate_normal(curandGenerator_t generator, float* output_ptr,
                                      std::size_t num, float mean, float stddev) {
    return curandGenerateNormal(generator, output_ptr, num, mean, stddev);
}

inline curandStatus_t generate_normal(curandGenerator_t generator, double* output_ptr,
                                      std::size_t num, double mean, double stddev) {
    return curandGenerateNormalDouble(generator, output_ptr, num, mean, stddev);
}

} // namespace spareribs
