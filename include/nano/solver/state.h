#pragma once

#include <nano/function.h>
#include <nano/solver/status.h>
#include <nano/solver/track.h>

namespace nano
{
///
/// \brief models a state (step) in a numerical optimization method.
///
/// NB: it handles both smooth and non-smooth with or without constraints.
///
class NANO_PUBLIC solver_state_t
{
public:
    ///
    /// \brief constructor
    ///
    solver_state_t(const function_t&, vector_t x0);

    ///
    /// \brief enable moving.
    ///
    solver_state_t(solver_state_t&&)                     = default;
    solver_state_t& operator=(solver_state_t&&) noexcept = default;

    ///
    /// \brief enable copying.
    ///
    solver_state_t(const solver_state_t&);
    solver_state_t& operator=(const solver_state_t&);

    ///
    /// \brief move to another point and returns true if the new point is valid.
    ///
    /// NB: optionally the Lagrangian multipliers for the equality and the inequality constraints can be given as well
    /// (if applicable).
    ///
    template <class tvector>
    bool update(const tvector& x, vector_cmap_t multiplier_equalities = vector_cmap_t{},
                vector_cmap_t multiplier_inequalities = vector_cmap_t{})
    {
        assert(x.size() == m_x.size());
        m_x  = x;
        m_fx = m_function(m_x, m_gx);
        return update(m_x, m_gx, m_fx, multiplier_equalities, multiplier_inequalities);
    }

    bool update(vector_cmap_t x, vector_cmap_t gx, scalar_t fx, vector_cmap_t multiplier_equalities = vector_cmap_t{},
                vector_cmap_t multiplier_inequalities = vector_cmap_t{});

    ///
    /// \brief update the number of function value and gradient evaluations.
    ///
    void update_calls();

    ///
    /// \brief try to update the current state and
    ///     returns true if the given function value is smaller than the current one.
    ///
    /// NB: this is usually called by non-monotonic solvers (e.g. for some non-smooth unconstrained optimization
    /// problems).
    ///
    bool update_if_better(const vector_t& x, scalar_t fx);
    bool update_if_better(const vector_t& x, const vector_t& gx, scalar_t fx);

    ///
    /// \brief update history of updates.
    ///
    void update_history();

    ///
    /// \brief convergence criterion of the function value:
    ///     no improvement in function value and parameter in the most recent updates.
    ///
    /// NB: appropriate for non-monotonic solvers (usually non-smooth problems) that call `update_if_better`.
    /// NB: this criterion is not theoretically motivated.
    ///
    scalar_t value_test(tensor_size_t patience) const;

    ///
    /// \brief convergence criterion of the gradient: the gradient magnitude relative to the function value.
    ///
    /// NB: only appropriate for smooth and unconstrained problems.
    ///
    scalar_t gradient_test() const;
    scalar_t gradient_test(vector_cmap_t gx) const;

    ///
    /// \brief return the KKT optimality criterion for constrained optimization:
    /// see (1) ch.5 "Convex Optimization", by S. Boyd and L. Vandenberghe, 2004.
    ///
    /// test 1: g_i(x) <= 0 (inequalities satisfied)
    /// test 2: h_j(x) == 0 (equalities satisfied)
    /// test 3: lambda_i >= 0 (positive multipliers for the inequalities)
    /// test 4: lambda_i * g_i(x) == 0
    /// test 5: grad(f(x)) + sum(lambda_i * grad(g_i(x))) + sum(miu_j * h_j(x)) == 0
    //
    /// NB: the optimality test is the maximum of the infinite norm of the 5 vector conditions.
    /// NB: only appropriate for constrained smooth problems.
    ///
    scalar_t kkt_optimality_test1() const;
    scalar_t kkt_optimality_test2() const;
    scalar_t kkt_optimality_test3() const;
    scalar_t kkt_optimality_test4() const;
    scalar_t kkt_optimality_test5() const;
    scalar_t kkt_optimality_test() const;

    ///
    /// \brief feasability test: the maximum deviation across all equality and inequality constraints
    /// as given by the first two KKT optimality conditions.
    ///
    scalar_t feasibility_test() const;

    ///
    /// \brief returns true if the current state is valid (e.g. no divergence is detected).
    ///
    bool valid() const;

    ///
    /// \brief returns the dot product between the gradient and the descent direction.
    ///
    /// NB: only appropriate for smooth problems.
    ///
    scalar_t dg(const vector_t& descent) const { return m_gx.dot(descent); }

    ///
    /// \brief returns true if the chosen direction is a descent direction.
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_descent(const vector_t& descent) const { return dg(descent) < 0.0; }

    ///
    /// \brief check if the current step satisfies the Armijo condition (sufficient decrease).
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_armijo(const solver_state_t& origin, const vector_t& descent, scalar_t step_size, scalar_t c1) const;

    ///
    /// \brief check if the current step satisfies the approximate Armijo condition (sufficient decrease).
    ///     see CG_DESCENT
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_approx_armijo(const solver_state_t& origin, scalar_t epsilon) const;

    ///
    /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature).
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_wolfe(const solver_state_t& origin, const vector_t& descent, scalar_t c2) const;

    ///
    /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature).
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_strong_wolfe(const solver_state_t& origin, const vector_t& descent, scalar_t c2) const;

    ///
    /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature).
    ///     see CG_DESCENT
    ///
    /// NB: only appropriate for smooth problems.
    ///
    bool has_approx_wolfe(const solver_state_t& origin, const vector_t& descent, scalar_t c1, scalar_t c2) const;

    ///
    /// \brief set the optimization status.
    ///
    void status(solver_status);

    ///
    /// \brief returns the (sub-)gradient.
    ///
    scalar_t fx() const { return m_fx; }

    ///
    /// \brief returns the current optimum parameter value.
    ///
    const vector_t& x() const { return m_x; }

    ///
    /// \brief returns the (sub-)gradient.
    ///
    const vector_t& gx() const { return m_gx; }

    ///
    /// \brief returns the number of function evaluation calls registered so far.
    ///
    tensor_size_t fcalls() const { return m_fcalls; }

    ///
    /// \brief returns the number of function gradient calls registered so far.
    ///
    tensor_size_t gcalls() const { return m_gcalls; }

    ///
    /// \brief returns the optimization status.
    ///
    solver_status status() const { return m_status; }

    ///
    /// \brief returns the function to minimize.
    ///
    const function_t& function() const { return m_function; }

    ///
    /// \brief returns the value of the equality constraints (if any).
    ///
    const vector_t& ceq() const { return m_ceq; }

    ///
    /// \brief returns the value of the inequality constraints (if any).
    ///
    const vector_t& cineq() const { return m_cineq; }

private:
    void update_constraints();

    // attributes
    const function_t& m_function;  ///< objective
    vector_t          m_x;         ///< parameter
    vector_t          m_gx;        ///< gradient
    scalar_t          m_fx{0};     ///< function value
    vector_t          m_ceq;       ///< equality constraint values
    vector_t          m_cineq;     ///< inequality constraint values
    vector_t          m_meq;       ///< Lagrange multiplies for equality constraints
    vector_t          m_mineq;     ///< Lagrange multiplies for inequality constraints
    vector_t          m_lgx;       ///< gradient of the Lagrangian dual function
    solver_status     m_status{};  ///< optimization status
    tensor_size_t     m_fcalls{0}; ///< number of function value evaluations so far
    tensor_size_t     m_gcalls{0}; ///< number of function gradient evaluations so far
    solver_track_t    m_track;     ///< history of updates
};

///
/// \brief pretty print the given solver state.
///
NANO_PUBLIC std::ostream& operator<<(std::ostream&, const solver_state_t&);
} // namespace nano
