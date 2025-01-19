#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over the probability simplex:
///     min  c.dot(x)
///     s.t. 1.dot(x) = 1, x >= 0.
///
class NANO_PUBLIC linear_program_cvx48d_eq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48d_eq_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};

///
/// \brief test/benchmark linear program from
///     exercise 4.8 (d), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over the probability simplex:
///     min  c.dot(x)
///     s.t. 1.dot(x) <= 1, x >= 0.
///
class NANO_PUBLIC linear_program_cvx48d_ineq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48d_ineq_t(tensor_size_t dims = 10);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
};
} // namespace nano
