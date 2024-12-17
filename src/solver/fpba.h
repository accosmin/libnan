#pragma once

#include <nano/solver.h>
#include <solver/bundle/nesterov.h>

namespace nano
{
///
/// \brief fast proximal bundle algorithms.
///
/// see (1) "Proximal bundle algorithms for nonsmooth convex optimization via fast gradient smooth methods",
///         by Ouorou, 2020
/// see (2) "Fast proximal algorithms for nonsmooth convex optimization", by Ouorou, 2020
/// see (3) "Adaptive restart for accelerated gradient schemes", by O'Donoghue, Candes, 2013
///
/// FIXME: describe bundle management, curve-search and proximity parameter strategy.
///
/// NB: the momentum for the accelerated schemes is reset adaptively using the function value scheme from (3).
///

///
/// \brief base class for fast proximal bundle algorithsm (FPBAx).
///
template <class tsequence>
class NANO_PUBLIC base_solver_fpba_t final : public solver_t
{
public:
    ///
    /// \brief constructor
    ///
    explicit base_solver_fpba_t();

    ///
    /// \brief @see clonable_t
    ///
    rsolver_t clone() const override;

    ///
    /// \brief @see solver_t
    ///
    solver_state_t do_minimize(const function_t&, const vector_t& x0, const logger_t&) const override;
};

///
/// \brief FPBA1 and FPBA2 from (1).
///
using solver_fpba1_t = base_solver_fpba_t<nesterov_sequence1_t>;
using solver_fpba2_t = base_solver_fpba_t<nesterov_sequence2_t>;
} // namespace nano
