#pragma once

#include <cassert>
#include <concepts>
#include <type_traits>

namespace spareribs {

template <std::unsigned_integral I1, std::unsigned_integral I2>
std::common_type_t<I1, I2> div_ceil(I1 dividend, I2 divisor) {
    using Result = std::common_type_t<I1, I2>;
    assert(divisor > I2{0});
    return (static_cast<Result>(dividend) + static_cast<Result>(divisor) - Result{1}) /
           static_cast<Result>(divisor);
}

} // namespace spareribs
