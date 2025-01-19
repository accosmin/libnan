#pragma once

#include <cstdint>

namespace nano
{
enum class convexity : uint8_t
{
    ignore,
    yes,
    no
};

enum class smoothness : uint8_t
{
    ignore,
    yes,
    no
};

enum class constrained : uint8_t
{
    ignore,
    yest,
    no
};
} // namespace nano
