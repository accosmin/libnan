#pragma once

#include <nano/function/linear.h>

namespace nano
{
///
/// \brief test/benchmark linear program from
///     exercise 4.10, see "Convex Optimization", by S. Boyd and L. Vanderberghe
///
/// standard form linear program:
///  min  c.dot(x)
///  s.t. Ax = b, x >= 0
///  and  the linear equality has exactly one solution (in this case A = D^t * D + I).
///
/// NB: b is generated as A * x, where x has all positive components (thus a feasible program)
///     if `feasible` is true, otherwise x has some negative components (thus an unfeasible program).
///
class NANO_PUBLIC linear_program_cvx410_t final : public linear_program_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit linear_program_cvx410_t(tensor_size_t dims = 10, bool feasible = true);

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
