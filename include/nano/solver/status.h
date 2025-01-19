#pragma once

#include <nano/enum.h>

namespace nano
{
enum class solver_status : uint8_t
{
    ///< maximum number of iterations reached without convergence (default)
    max_iters,

    ///< optimization failed (e.g. line-search failed)
    failed,

    ///< cannot find any feasible point (if constrained).
    unfeasible,

    ///< problem is not lower-bounded
    unbounded,

    ///< incompatible function to minimize (e.g. solver only supports linear and quadratic programs)
    incompatible,

    ///< no theoretical motivated stopping criterion.
    ///< heuristically the algorithm stops when no sufficient decrease in the most recent number of iterations.
    ///< applicable to all problems: convex/non-convex, smooth/non-smooth, w/o constraints.
    value_test,

    ///< theoretical motivated stopping criterion: the relative magnitude of the gradient.
    ///< applicable only to smooth problems convex and non-convex, but without constraints.
    gradient_test,

    ///< theoretical motivated stopping criterion specific to a particular algorithm (and a class of functions).
    ///< e.g. an upper bound of the gap between the current point and the optimum.
    specific_test,

    ///< theoretical motivated stopping criterion: the KKT optimality conditions.
    ///< applicable only to constrained problems, but mostly useful for convex smooth constrained problems.
    kkt_optimality_test,
};

template <>
inline enum_map_t<solver_status> enum_string<solver_status>()
{
    return {
        {          solver_status::max_iters,           "max_iters"},
        {             solver_status::failed,              "failed"},
        {         solver_status::unfeasible,          "unfeasible"},
        {          solver_status::unbounded,           "unbounded"},
        {       solver_status::incompatible,        "incompatible"},
        {         solver_status::value_test,          "value-test"},
        {      solver_status::gradient_test,       "gradient-test"},
        {      solver_status::specific_test,       "specific-test"},
        {solver_status::kkt_optimality_test, "kkt-optimality-test"}
    };
}
} // namespace nano
