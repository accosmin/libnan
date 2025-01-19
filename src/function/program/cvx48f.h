#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.8 (f), see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize a linear function over a unit box with a weighted budget constraint:
///  min  c.dot(x)
///  s.t. d.dot(x) = alpha * 1.dot(d), 0 <= x <= 1
///  and  d > 0 and 0 <= alpha <= 1.
///
class NANO_PUBLIC linear_program_cvx48f_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx48f_t(tensor_size_t dims = 10, scalar_t alpha = 0.5);

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
