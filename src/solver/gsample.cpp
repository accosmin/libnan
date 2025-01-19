#include <solver/gsample.h>
#include <solver/gsample/lsearch.h>
#include <solver/gsample/preconditioner.h>
#include <solver/gsample/sampler.h>

using namespace nano;

template <class tsampler, class tpreconditioner>
base_solver_gs_t<tsampler, tpreconditioner>::base_solver_gs_t()
    : solver_t(scat(tsampler::str(), tpreconditioner::str()))
{
    const auto basename = scat("solver::", type_id(), "::");

    register_parameter(parameter_t::make_scalar(basename + "miu0", 0, LE, 1e-6, LT, 1e+6));
    register_parameter(parameter_t::make_scalar(basename + "epsilon0", 0, LT, 0.1, LT, 1e+6));
    register_parameter(parameter_t::make_scalar(basename + "theta_miu", 0, LT, 1.0, LE, 1));
    register_parameter(parameter_t::make_scalar(basename + "theta_epsilon", 0, LT, 0.1, LE, 1));

    register_parameter(parameter_t::make_scalar(basename + "lsearch_beta", 0, LE, 1e-8, LT, 1));
    register_parameter(parameter_t::make_scalar(basename + "lsearch_gamma", 0, LT, 0.5, LT, 1));
    register_parameter(parameter_t::make_scalar(basename + "lsearch_perturb_c", 0, LE, 1e-6, LT, 1));
    register_parameter(parameter_t::make_integer(basename + "lsearch_max_iters", 0, LT, 50, LE, 100));
}

template <class tsampler, class tpreconditioner>
rsolver_t base_solver_gs_t<tsampler, tpreconditioner>::clone() const
{
    return std::make_unique<base_solver_gs_t<tsampler, tpreconditioner>>(*this);
}

template <class tsampler, class tpreconditioner>
solver_state_t base_solver_gs_t<tsampler, tpreconditioner>::do_minimize(const function_t& function, const vector_t& x0,
                                                                        const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto basename      = scat("solver::", type_id(), "::");
    const auto max_evals     = parameter("solver::max_evals").template value<tensor_size_t>();
    const auto epsilon       = parameter("solver::epsilon").template value<scalar_t>();
    const auto miu0          = parameter(basename + "miu0").template value<scalar_t>();
    const auto epsilon0      = parameter(basename + "epsilon0").template value<scalar_t>();
    const auto theta_miu     = parameter(basename + "theta_miu").template value<scalar_t>();
    const auto theta_epsilon = parameter(basename + "theta_epsilon").template value<scalar_t>();

    const auto n = function.size();

    auto x        = vector_t{n};
    auto g        = vector_t{n};
    auto miuk     = miu0;
    auto epsilonk = epsilon0;
    auto state    = solver_state_t{function, x0};
    auto sampler  = tsampler{n};
    auto precond  = tpreconditioner{n};
    auto lsearch  = gsample::lsearch_t{n, *this, basename};

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // sample gradients within the given radius
        sampler.sample(state, epsilonk);

        // update preconditioner
        precond.update(sampler, state, epsilonk);

        // solve the quadratic problem to find the stabilized gradient
        sampler.descent(precond.W(), g, logger);

        // check convergence
        const auto iter_ok   = g.all_finite() && epsilonk > std::numeric_limits<scalar_t>::epsilon();
        const auto converged = state.gradient_test(g) < epsilon && epsilonk < epsilon;
        if (solver_t::done_specific_test(state, iter_ok, converged, logger))
        {
            break;
        }

        // too small gradient, reduce sampling radius (potentially convergence detected)
        else if (const auto gnorm2 = g.lpNorm<2>(); gnorm2 <= miuk)
        {
            precond.update(1.0);

            miuk *= theta_miu;
            epsilonk *= theta_epsilon;
        }

        // line-search step
        else
        {
            const auto alphak = lsearch.step(x, g, state, precond.H());
            precond.update(alphak);

            /*if (alphak < std::numeric_limits<scalar_t>::epsilon())
            {
                // NB: line-search failed, reduce the sampling radius - see (1)
                // epsilonk *= theta_epsilon;
            }*/
        }
    }

    state.update_calls();
    return state;
}

template class nano::base_solver_gs_t<gsample::fixed_sampler_t, gsample::identity_preconditioner_t>;
template class nano::base_solver_gs_t<gsample::adaptive_sampler_t, gsample::identity_preconditioner_t>;

template class nano::base_solver_gs_t<gsample::fixed_sampler_t, gsample::lbfgs_preconditioner_t>;
template class nano::base_solver_gs_t<gsample::adaptive_sampler_t, gsample::lbfgs_preconditioner_t>;
