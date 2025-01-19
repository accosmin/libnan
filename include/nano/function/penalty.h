#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief base penalty function.
///
class NANO_PUBLIC penalty_function_t : public function_t
{
public:
    ///
    /// \brief default constructor
    ///
    explicit penalty_function_t(const function_t&, const char* prefix);

    ///
    /// \brief set the penalty term.
    ///
    penalty_function_t& penalty(scalar_t);

    ///
    /// \brief returns the penalty term.
    ///
    scalar_t penalty() const { return m_penalty; }

    ///
    /// \brief returns the original constrained optimization objective.
    ///
    const function_t& function() const { return m_function; }

private:
    // attributes
    const function_t& m_function;     ///<
    scalar_t          m_penalty{1.0}; ///<
};

///
/// \brief (exact) linear penalty function:
///     q(c, x) = f(x) + c * sum(|h_j(x)|, j) + c * sum(max(0, g_i(x)), i),
///
/// where:
///     * f(x) is the objective function to minimize,
///     * c > 0 is the penalty term - the higher the better the penalty function approximates
///         the original constrained optimization problem,
///     * {h_j(x) == 0} is the set of equality constraints and
///     * {g_i(x) <= 0} is the set of inequality constraints.
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: the penalty is exact: the penalty term doesn't need to be increased to infinity to obtain an exact solution.
/// NB: the penalty is non-smooth and as such line-search solvers cannot be used.
///
class NANO_PUBLIC linear_penalty_function_t final : public penalty_function_t
{
public:
    ///
    /// \brief default constructor
    ///
    explicit linear_penalty_function_t(const function_t&);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;
};

///
/// \brief quadratic penalty function:
///     q(c, x) = f(x) + c * sum(h_j(x)^2, j) + c * sum(max(0, g_i(x))^2, i),
///
/// where:
///     * f(x) is the objective function to minimize,
///     * c > 0 is the penalty term - the higher the better the penalty function approximates
///         the original constrained optimization problem,
///     * {h_j(x) == 0} is the set of equality constraints and
///     * {g_i(x) <= 0} is the set of inequality constraints.
///
/// see "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
/// NB: the penalty is not exact: the penalty term needs to be increased to infinity to obtain an exact solution.
/// NB: the penalty is continuous and differentiable (not necessarily C^1), thus usable with line-search solvers.
///
class NANO_PUBLIC quadratic_penalty_function_t final : public penalty_function_t
{
public:
    ///
    /// \brief default constructor
    ///
    explicit quadratic_penalty_function_t(const function_t&);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;
};

///
/// \brief augmented lagrangian function:
///     q(c, x) = f(x) + ro/2 * sum((h_j(x) + lambda_j/ro)^2, j) + ro/2 * sum(max(0, g_i(x) + miu_i/ro)^2, i),
///
/// where:
///     * f(x) is the objective function to minimize,
///     * c > 0 is the penalty term - the higher the better the penalty function approximates
///         the original constrained optimization problem,
///     * lambda and miu are approximations of the Lagrange multipliers approximation associated to constraints,
///     * {h_j(x) == 0} is the set of equality constraints and
///     * {g_i(x) <= 0} is the set of inequality constraints.
///
/// see "Practical Augmented Lagrangian Methods", by E. G. Birgin, J. M. Martinez, 2007.
///
/// NB: the penalty is continuous and differentiable (not necessarily C^1), thus usable with line-search solvers.
///
class NANO_PUBLIC augmented_lagrangian_function_t final : public penalty_function_t
{
public:
    ///
    /// \brief default constructor
    ///
    explicit augmented_lagrangian_function_t(const function_t&, const vector_t& lambda, const vector_t& miu);

    ///
    /// \brief @see clonable_t
    ///
    rfunction_t clone() const override;

    ///
    /// \brief @see function_t
    ///
    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override;

private:
    // attributes
    const vector_t& m_lambda; ///< approximations of the Lagrange multipliers for equality constraints
    const vector_t& m_miu;    ///< approximations of the Lagrange multipliers for inequality constraints
};
} // namespace nano
