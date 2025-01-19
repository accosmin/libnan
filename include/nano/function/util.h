#pragma once

#include <nano/function.h>

namespace nano
{
///
/// \brief compute the gradient accuracy (given vs. central finite difference approximation).
///
NANO_PUBLIC scalar_t grad_accuracy(const function_t&, const vector_t& x, scalar_t desired_epsilon = 0.0);

///
/// \brief check if the function is convex along the [x1, x2] line.
///
NANO_PUBLIC bool is_convex(const function_t&, const vector_t& x1, const vector_t& x2, int steps,
                           scalar_t epsilon = epsilon1<scalar_t>());

///
/// \brief return true if the equality constraint `Ax = b` is not full row rank.
///
/// in this case the constraints are transformed in-place to obtain row-independant linear constraints
///     by performing an appropriate matrix decomposition.
///
NANO_PUBLIC bool reduce(matrix_t& A, vector_t& b);

///
/// \brief returns true whether the given quadratic term is convex (aka positive semi-definite).
///
NANO_PUBLIC bool is_convex(const matrix_t&, scalar_t tol = 1e-9);

///
/// \brief returns the strong convexity factor of a quadratic term, or zero if not convex.
///
NANO_PUBLIC scalar_t strong_convexity(const matrix_t&);

///
/// \brief return a strictly feasible point wrt the given inequality constraints `Ax <= b`, if possible.
///
NANO_PUBLIC std::optional<vector_t> make_strictly_feasible(const matrix_t& A, const vector_t& b);

///
/// \brief return a compact linear representation (A, b, G, h) of the functional constraints (if possible):
///     Ax = b (gathers all equality constraints) and
///     Gx <= b (gathers all inequality constraints).
///
/// NB: if any constraint is not linear, then std::nullopt is returned.
///
struct linear_constraints_t
{
    matrix_t m_A; ///<
    vector_t m_b; ///<
    matrix_t m_G; ///<
    vector_t m_h; ///<
};

NANO_PUBLIC std::optional<linear_constraints_t> make_linear_constraints(const function_t&);
} // namespace nano
