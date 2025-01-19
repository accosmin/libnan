#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.9, see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// minimize the square linear program:
///  min  c.dot(x)
///  s.t. Ax <= b
///  and  A is square and nonsingular and A^T * c <= 0 (to be feasible).
///
class NANO_PUBLIC linear_program_cvx49_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx49_t(tensor_size_t dims = 10);

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
