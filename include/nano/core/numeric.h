#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

namespace nano
{
///
/// \brief square: x^2.
///
template <class tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
tscalar square(const tscalar value) noexcept
{
    return value * value;
}

///
/// \brief cube: x^3.
///
template <class tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
tscalar cube(const tscalar value) noexcept
{
    return value * square(value);
}

///
/// \brief quartic: x^4.
///
template <class tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
tscalar quartic(const tscalar value) noexcept
{
    return square(square(value));
}

///
/// \brief integer division with rounding.
///
template <class tnominator, class tdenominator, std::enable_if_t<std::is_integral_v<tnominator>, bool> = true,
          std::enable_if_t<std::is_integral_v<tdenominator>, bool> = true>
tnominator idiv(const tnominator nominator, const tdenominator denominator) noexcept
{
    return (nominator + static_cast<tnominator>(denominator) / 2) / static_cast<tnominator>(denominator);
}

///
/// \brief integer rounding.
///
template <class tvalue, class tmodulo, std::enable_if_t<std::is_integral_v<tvalue>, bool> = true,
          std::enable_if_t<std::is_integral_v<tmodulo>, bool> = true>
tvalue iround(const tvalue value, const tmodulo modulo) noexcept
{
    return idiv(value, modulo) * modulo;
}

///
/// \brief check if two scalars are almost equal.
///
template <class tscalar1, class tscalar2, std::enable_if_t<std::is_arithmetic_v<tscalar1>, bool> = true,
          std::enable_if_t<std::is_arithmetic_v<tscalar2>, bool> = true>
bool close(const tscalar1 lhs, const tscalar2 rhs, const double epsilon) noexcept
{
    return std::fabs(static_cast<double>(lhs) - static_cast<double>(rhs)) <
           epsilon * (1.0 + (std::fabs(static_cast<double>(lhs)) + std::fabs(static_cast<double>(rhs)) / 2));
}

///
/// \brief round to the closest power of 10.
///
template <class tscalar, std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true>
inline tscalar roundpow10(const tscalar v) noexcept
{
    return std::pow(tscalar(10), std::round(std::log10(v)));
}

///
/// \brief precision level [0=very precise, 1=quite precise, 2=precise, 3=loose] for different scalars.
///
template <class tscalar, std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true>
tscalar epsilon() noexcept
{
    return std::numeric_limits<tscalar>::epsilon();
}

template <class tscalar, std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true>
tscalar epsilon0() noexcept
{
    return roundpow10(10 * epsilon<tscalar>());
}

template <class tscalar, std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true>
tscalar epsilon1() noexcept
{
    const auto cb = std::cbrt(epsilon<tscalar>());
    return roundpow10(cb * cb);
}

template <class tscalar, std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true>
tscalar epsilon2() noexcept
{
    return roundpow10(std::sqrt(epsilon<tscalar>()));
}

template <class tscalar, std::enable_if_t<std::is_floating_point_v<tscalar>, bool> = true>
tscalar epsilon3() noexcept
{
    return roundpow10(std::cbrt(epsilon<tscalar>()));
}

///
/// \brief check if the given scalar is finite.
/// NB: handles explicitly integer values as MSVC doesn't cast it to the appropriate floating point types.
///
template <class tscalar, std::enable_if_t<std::is_arithmetic_v<tscalar>, bool> = true>
bool isfinite([[maybe_unused]] const tscalar value) noexcept
{
    if constexpr (std::is_floating_point_v<tscalar>)
    {
        return std::isfinite(value);
    }
    else
    {
        return true;
    }
}
} // namespace nano
