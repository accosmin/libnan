#pragma once

#include <nano/solver.h>

namespace nano
{
class penalty_function_t;

///
/// \brief base class for exterior penalty methods.
///
class NANO_PUBLIC solver_penalty_t : public solver_t
{
public:
    using solver_t::minimize;

    ///
    /// \brief constructor
    ///
    explicit solver_penalty_t(string_t id);

    ///
    /// \brief minimize the given penalty function starting from the initial point x0.
    ///
    solver_state_t minimize(penalty_function_t&, const vector_t& x0, const logger_t&) const;
};

///
/// \brief exterior penalty method using the linear penalty function.
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
///
/// NB: the penalty method works by increasing the penalty term in the outer loop
///     and using the given solver to minimize the penalty function (the inner loop).
///
/// NB: the penalty function is exact,
///     but it is not smooth and thus the optimization is typically not very precise.
///
class NANO_PUBLIC solver_linear_penalty_t final : public solver_penalty_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_linear_penalty_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;
};

///
/// \brief exterior penalty method using the quadratic penalty function.
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006
///
/// NB: the penalty method works by increasing the penalty term in the outer loop
///     and using the given solver to minimize the penalty function (the inner loop).
///
/// NB: the penalty function is not exact,
///     but it is smooth and thus the optimization is typically very precise.
///
class NANO_PUBLIC solver_quadratic_penalty_t final : public solver_penalty_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_quadratic_penalty_t();

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
