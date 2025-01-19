#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief models the class of linear programs.
///
/// general form (equality constraint, inequality constraint):
///     min  c.dot(x)
///     s.t. A * x = b
///     and  G * x <= h.
///
/// standard form (equality constraint, no inequality constraint):
///     min  c.dot(x)
///     s.t. A * x = b
///     and  x >= 0.0.
///
/// inequality form (no equality constraint, inequality constraint):
///     min  c.dot(x)
///     s.t. A * x <= b.
///
/// rectangle-inequality form (no equality constraint, inequality constraint):
///     min  c.dot(x)
///     s.t. l <= x <= u.
///
/// NB: the equality and the inequality constraints are optional.
///
/// see (1) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
/// see (2) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
class NANO_PUBLIC linear_program_t : public function_t
{
public:
    ///
    /// \brief constructor
    ///
    linear_program_t(string_t id, vector_t c);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

    ///
    /// \brief @see function_t
    ///
    bool constrain(constraint_t&&) override;

    ///
    /// \brief return the objective's parameters (need explicitly by some solvers).
    ///
    const vector_t& c() const { return m_c; }

    ///
    /// \brief change the objective with a compatible one and keep the constraints.
    ///
    void reset(vector_t c);

private:
    // attributes
    vector_t m_c; ///<
};
} // namespace nano
