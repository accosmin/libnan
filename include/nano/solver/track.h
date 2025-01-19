#pragma once

#include <nano/tensor.h>

namespace nano
{
///
/// \brief track the point updates of some solver's iterates. this is useful to heuristically stop the optimization
/// when no progress has been made in the recent iterations.
///
/// NB: useful for non-smooth optimization with some solvers that don't have a proper stopping criterion.
///
/// NB: useful for constrained optimization with some solvers that don't produce Lagrange multipliers and thus
/// no KKT optimality criterion cannot be computed.
///
class solver_track_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_track_t(vector_t x, scalar_t fx);

    ///
    /// \brief update the current iteration.
    ///
    void update(vector_t x, scalar_t fx);

    ///
    /// \brief convergence criterion of the function value for unconstrained problems.
    ///
    scalar_t value_test_unconstrained(tensor_size_t patience) const;

    ///
    /// \brief convergence criterion of the function value for constrained problems.
    ///
    scalar_t value_test_constrained(tensor_size_t patience) const;

private:
    using scalars_t = std::vector<scalar_t>;

    // attributes
    vector_t  m_prev_x;     ///< previous point
    scalar_t  m_prev_fx;    ///< previous function value
    scalars_t m_history_dx; ///< history of point differences across iterates
    scalars_t m_history_df; ///< history of function value differences across iterates
};
}
