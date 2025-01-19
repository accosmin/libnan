#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief models the optimum minimum of a function (the solution of a numerical optimization problem).
///
struct optimum_t
{
    enum class status : uint8_t
    {
        solvable,
        unfeasible,
        unbounded,
    };

    static constexpr auto NaN = std::numeric_limits<scalar_t>::quiet_NaN();

    // attributes
    vector_t m_xbest;                    ///< optimum: solution
    scalar_t m_fbest{NaN};               ///< optimum: function value
    status   m_status{status::solvable}; ///< expected convergence status
};
} // namespace nano
