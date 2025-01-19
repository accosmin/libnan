#include <nano/solver/csearch.h>
#include <nano/solver/proximity.h>
#include <solver/rqb.h>

using namespace nano;

solver_rqb_t::solver_rqb_t()
    : solver_t("rqb")
{
    const auto prefix = string_t{"solver::rqb"};
    bundle_t::config(*this, prefix);
    csearch_t::config(*this, prefix);
    proximity_t::config(*this, prefix);
}

rsolver_t solver_rqb_t::clone() const
{
    return std::make_unique<solver_rqb_t>(*this);
}

solver_state_t solver_rqb_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto prefix    = string_t{"solver::rqb"};
    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();

    auto state     = solver_state_t{function, x0};
    auto bundle    = bundle_t::make(state, *this, prefix);
    auto csearch   = csearch_t::make(function, *this, prefix);
    auto proximity = proximity_t::make(state, *this, prefix);

    auto Gn  = state.gx(); ///< approximation of the gradient of Moreau-Yosida regularization model at x_n
    auto Gn1 = state.gx(); ///< same at x_{n+1} - the next proximity center

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
            Gn1 = bundle.smeared_s();
            proximity.update(t, bundle.x(), y, bundle.gx(), gy, Gn, Gn1);
            Gn = Gn1;

            bundle.moveto(y, gy, fy);
            assert(fy < state.fx());
            state.update(y, gy, fy);
        }
        else if (status == csearch_status::cutting_plane_step)
        {
            Gn = bundle.smeared_s();

            bundle.moveto(y, gy, fy);
            assert(fy < state.fx());
            state.update(y, gy, fy);
        }
        else if (status == csearch_status::null_step)
        {
            bundle.append(y, gy, fy);
        }
    }

    state.update_calls();
    return state;
}
