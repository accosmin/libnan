#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) = alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
class NANO_PUBLIC linear_program_cvx48e_eq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48e_eq_t(tensor_size_t dims = 10, tensor_size_t alpha = 0);

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
///     exercise 4.8 (e), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a total budget constraint:
///  min  c.dot(x)
///  s.t. 1.dot(x) <= alpha, 0 <= x <= 1
///  and  alpha is an integer between 0 and n.
///
class NANO_PUBLIC linear_program_cvx48e_ineq_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48e_ineq_t(tensor_size_t dims = 10, tensor_size_t alpha = 0);

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
