#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.8 (b), see "Convex Optimization", by S. Boyd and L. Vanderberghe.
///
/// minimize a linear function over a halfspace:
///     min  c.dot(x)
///     s.t. a.dot(x) <= b,
///     s.t. c = lambda * a
///     and  lambda <= 0.0.
///
class NANO_PUBLIC linear_program_cvx48b_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48b_t(tensor_size_t dims = 10, scalar_t lambda = -1.0);

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
