#pragma once

#include <nano/solver.h>

namespace nano
{
///
/// \brief (practical) augmented lagrangian method.
///
/// see (1) "Practical Augmented Lagrangian Methods", by E. G. Birgin, J. M. Martinez, 2007.
/// see (2) "On Augmented Lagrangian Methods With General Lower-Level Constraints", by Andreani, Birgin, Martinez,
/// Schuverdt, 2008.
///
/// NB: the penalty parameter and the estimated lagrangian multipliers are updated at the same time
///     in a loop to solve a potentially smooth function - the augmented lagrangian
///     (continuous and differentiable, but not necessarily with continuous gradients).
///
/// NB: the augmented lagrangian is solved without any bounds constraints like in the original formulation.
///
class NANO_PUBLIC solver_augmented_lagrangian_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_augmented_lagrangian_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;
};
} // namespace nano
