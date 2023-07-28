#pragma once

#include <memory>
#include <nano/arch.h>
#include <nano/eigen.h>

namespace nano::linprog
{
///
/// \brief the standard form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and x >= 0.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
/// see (2) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
///
struct NANO_PUBLIC problem_t
{
    ///
    /// \brief constructor
    ///
    problem_t(vector_t c, matrix_t A, vector_t b);

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon) const;

    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
};

///
/// \brief the solution of the standard form of linear programming.
///
struct NANO_PUBLIC solution_t
{
    ///
    /// \brief returns true if convergence is detected.
    ///
    bool converged(scalar_t max_kkt_violation = 1e-16) const;

    // attributes
    static constexpr auto max = std::numeric_limits<scalar_t>::max();

    vector_t m_x;        ///< solution (primal problem)
    vector_t m_l;        ///< solution (dual problem) - equality constraints
    vector_t m_s;        ///< solution (dual problem) - inequality constraints
    int      m_iters{0}; ///< number of iterations
    scalar_t m_miu{max}; ///< duality measure: ~zero (converged), very large/infinite (not feasible, unbounded)
    scalar_t m_kkt{max}; ///< maximum deviation of the KKT sufficient conditions
};

///
/// \brief the inequality form of linear programming:
///     f(x) = c.dot(x) s.t Ax <= b.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
/// see (2) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
///
struct NANO_PUBLIC inequality_problem_t
{
    ///
    /// \brief constructor
    ///
    inequality_problem_t(vector_t c, matrix_t A, vector_t b);

    ///
    /// \brief return the equivalent standard form problem.
    ///
    problem_t transform() const;

    ///
    /// \brief return the equivalent solution from the given solution of the equivalent standard form problem.
    ///
    solution_t transform(const solution_t&) const;

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon) const;

    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
};

///
/// \brief the general form of linear programming:
///     f(x) = c.dot(x) s.t Ax = b and Gx <= h.
///
/// see (1) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
/// see (2) "Convex Optimization", by S. Boyd and L. Vanderberghe, 2004.
///
struct NANO_PUBLIC general_problem_t
{
    ///
    /// \brief constructor
    ///
    general_problem_t(vector_t c, matrix_t A, vector_t b, matrix_t G, vector_t h);

    ///
    /// \brief return the equivalent standard form problem.
    ///
    problem_t transform() const;

    ///
    /// \brief return the equivalent solution from the given solution of the equivalent standard form problem.
    ///
    solution_t transform(const solution_t&) const;

    ///
    /// \brief return true if the given point is feasible with the given threshold.
    ///
    bool feasible(const vector_t& x, scalar_t epsilon) const;

    // attributes
    vector_t m_c; ///<
    matrix_t m_A; ///<
    vector_t m_b; ///<
    matrix_t m_G; ///<
    vector_t m_h; ///<
};

using logger_t = std::function<void(const problem_t&, const solution_t&)>;

///
/// \brief parameters for the linear program solver.
///
struct NANO_PUBLIC params_t
{
    // attributes
    int      m_max_iters{100};     ///< maximum number of iterations
    scalar_t m_kkt_epsilon{1e-16}; ///< maximum deviation of the KKT conditions
    int      m_kkt_patience{3}; ///< maximum number of iterations to wait if the maximum KKT deviation doesn't improve
    logger_t m_logger{};        ///< logging callback
};

NANO_PUBLIC params_t make_params(logger_t logger = logger_t{});

///
/// \brief returns the solution of the given linear program using the predictor-corrector algorithm.
///
/// see (1) "On the implementation of a primal-dual interior point method", by S. Mehrotra, 1992.
/// see (2) ch.14 (page 411) "Numerical Optimization", by J. Nocedal, S. Wright, 2006.
///
NANO_PUBLIC solution_t solve(const problem_t&, const params_t& params = params_t{});
NANO_PUBLIC solution_t solve(const general_problem_t&, const params_t& params = params_t{});
NANO_PUBLIC solution_t solve(const inequality_problem_t&, const params_t& params = params_t{});
} // namespace nano::linprog
