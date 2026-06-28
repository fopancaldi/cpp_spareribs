#pragma once

#include <concepts>

namespace spareribs::concepts {

template <typename T>
concept gpu_float = std::same_as<T, float> or std::same_as<T, double>;

}
