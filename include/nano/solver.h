#pragma once

#include <nano/solver/lsearch.h>

namespace nano
{
class solver_t;
using rsolver_t  = std::unique_ptr<solver_t>;
using rsolvers_t = std::vector<rsolver_t>;

///
/// \brief interface for numerical optimization algorithms.
///
/// NB: the resulting point for the unconstrained case (if enough iterations have been used) is either:
///     - the global minimum if the function is convex or
///     - a critical point (not necessarily a local minimum) otherwise.
///
class NANO_PUBLIC solver_t : public typed_t, public configurable_t, public clonable_t<solver_t>
{
public:
    ///
    /// \brief constructor
    ///
    explicit solver_t(string_t id);

    ///
    /// \brief enable copying
    ///
    solver_t(const solver_t&);
    solver_t& operator=(const solver_t&) = delete;

    ///
    /// \brief enable moving
    ///
    solver_t(solver_t&&) noexcept   = default;
    solver_t& operator=(solver_t&&) = default;

    ///
    /// \brief destructor
    ///
    ~solver_t() override = default;

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<solver_t>& all();

    ///
    /// \brief minimize the given function starting from the initial point x0 until:
    ///     - convergence is achieved (e.g. critical point, possiblly a local/global minima) or
    ///     - the maximum number of iterations is reached or
    ///     - the solver failed (e.g. line-search failed).
    ///
    solver_state_t minimize(const function_t&, const vector_t& x0, const logger_t&) const;

    ///
    /// \brief set the line-search initialization method.
    ///
    void lsearch0(const lsearch0_t&);
    void lsearch0(const string_t& id);

    ///
    /// \brief set the line-search strategy method.
    ///
    void lsearchk(const lsearchk_t&);
    void lsearchk(const string_t& id);

    ///
    /// \brief change the solver to be more precise by the given factor in the range (0, 1).
    ///
    void more_precise(scalar_t epsilon_factor);

    ///
    /// \brief return the line-search initialization method.
    ///
    const lsearch0_t& lsearch0() const { return *m_lsearch0; }

    ///
    /// \brief return the the line-search strategy method.
    ///
    const lsearchk_t& lsearchk() const { return *m_lsearchk; }

protected:
    bool done_value_test(solver_state_t&, bool iter_ok, const logger_t&) const;
    bool done_gradient_test(solver_state_t&, bool iter_ok, const logger_t&) const;
    bool done_specific_test(solver_state_t&, bool iter_ok, bool converged, const logger_t&) const;
    bool done_kkt_optimality_test(solver_state_t&, bool iter_ok, const logger_t&) const;

    void warn_nonconvex(const function_t&, const logger_t&) const;
    void warn_nonsmooth(const function_t&, const logger_t&) const;
    void warn_constrained(const function_t&, const logger_t&) const;

    lsearch_t        make_lsearch() const;
    static rsolver_t make_solver(const function_t&, scalar_t epsilon, tensor_size_t max_evals);

    virtual solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const = 0;

private:
    bool done(solver_state_t&, bool iter_ok, solver_status, const logger_t&) const;

    // attributes
    rlsearch0_t m_lsearch0; ///<
    rlsearchk_t m_lsearchk; ///<
};
} // namespace nano
