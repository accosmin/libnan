#pragma once

#include <nano/solver.h>
#include <solver/interior/program.h>

namespace nano
{
///
/// \brief primal-dual interior-point solver specialized for linear and quadratic programs.
///
/// see (1) ch.5,6 "Primal-dual interior-point methods", by S. Wright, 1997.
/// see (2) ch.11 "Convex Optimization", by S. Boyd and L. Vandenberghe, 2004.
/// see (3) ch.14,16,19 "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
class NANO_PUBLIC solver_ipm_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_ipm_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;

private:
    solver_state_t do_minimize(const program_t&, const vector_t& x0, const logger_t&) const;
    solver_state_t do_minimize_with_inequality(const program_t&, const vector_t& x0, const logger_t&) const;
    solver_state_t do_minimize_without_inequality(const program_t&, const vector_t& x0, const logger_t&) const;
};
} // namespace nano
