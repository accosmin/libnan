#include <nano/solver/bundle.h>
#include <nano/solver/csearch.h>
#include <nano/solver/proximity.h>
#include <solver/fpba.h>

using namespace nano;

template <class tsequence>
base_solver_fpba_t<tsequence>::base_solver_fpba_t()
    : solver_t(scat("fpba", tsequence::str()))
{
    const auto prefix = scat("solver::", type_id());
    bundle_t::config(*this, prefix);
    csearch_t::config(*this, prefix);
    proximity_t::config(*this, prefix);
}

template <class tsequence>
rsolver_t base_solver_fpba_t<tsequence>::clone() const
{
    return std::make_unique<base_solver_fpba_t<tsequence>>(*this);
}

template <class tsequence>
solver_state_t base_solver_fpba_t<tsequence>::do_minimize(const function_t& function, const vector_t& x0,
                                                          const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto prefix    = scat("solver::", type_id());
    const auto max_evals = parameter("solver::max_evals").template value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").template value<scalar_t>();

    auto state     = solver_state_t{function, x0};
    auto bundle    = bundle_t::make(state, *this, prefix);
    auto csearch   = csearch_t::make(function, *this, prefix);
    auto proximity = proximity_t::make(state, *this, prefix);

    auto gx       = vector_t{x0.size()};
    auto sequence = tsequence{state};

    const auto apply_nesterov_sequence = [&](const vector_t& z, const vector_t& gz, const scalar_t fz)
    {
        state.update_if_better(z, gz, fz);

        // nesterov's momentum on the proximity center
        const auto& x  = sequence.update(z);
        const auto  fx = function(x, gx);
        bundle.moveto(x, gx, fx);

        // update best point and reset momentum if no improvement
        if (!state.update_if_better(x, gx, fx))
        {
            sequence.reset();
        }
    };

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto& [t, status, y, gy, fy] = csearch.search(bundle, proximity.miu(), max_evals, epsilon, logger);

        const auto iter_ok   = status != csearch_status::failed;
        const auto converged = status == csearch_status::converged;
        if (solver_t::done_specific_test(state, iter_ok, converged, logger))
        {
            break;
        }

        if (status == csearch_status::descent_step)
        {
            proximity.update(t, bundle.x(), y, bundle.gx(), gy);
            apply_nesterov_sequence(y, gy, fy);
        }
        else if (status == csearch_status::cutting_plane_step)
        {
            apply_nesterov_sequence(y, gy, fy);
        }
        else if (status == csearch_status::null_step)
        {
            bundle.append(y, gy, fy);
        }
    }

    state.update_calls();
    return state;
}

template class nano::base_solver_fpba_t<nesterov_sequence1_t>;
template class nano::base_solver_fpba_t<nesterov_sequence2_t>;
