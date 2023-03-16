#include <nano/solver/nonsmooth_state.h>
#include <nano/solver/sgm.h>

using namespace nano;

solver_sgm_t::solver_sgm_t()
    : solver_t("sgm")
{
    type(solver_type::non_monotonic);

    register_parameter(parameter_t::make_scalar("solver::sgm::power", 0.5, LE, 0.75, LT, 1.0));
    register_parameter(parameter_t::make_integer("solver::sgm::patience", 10, LE, 100, LE, 1e+6));
}

rsolver_t solver_sgm_t::clone() const
{
    return std::make_unique<solver_sgm_t>(*this);
}

solver_state_t solver_sgm_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto power     = parameter("solver::sgm::power").value<scalar_t>();
    const auto patience  = parameter("solver::sgm::patience").value<tensor_size_t>();

    auto state = solver_state_t{function, x0}; // best state
    auto track = nonsmooth_solver_state_t{state, patience};

    auto x = state.x;
    auto g = state.g;

    auto iteration = 0;
    while (function.fcalls() < max_evals)
    {
        const auto gnorm = g.lpNorm<Eigen::Infinity>();
        if (gnorm < std::numeric_limits<scalar_t>::epsilon())
        {
            const auto iter_ok   = true;
            const auto converged = true;
            solver_t::done(function, state, iter_ok, converged);
            break;
        }

        const auto lambda = 1.0 / std::pow(iteration + 1, power);
        x -= lambda * g / gnorm;

        const auto f = function.vgrad(x, &g);
        track.update_if_better(x, g, f);

        const auto iter_ok   = std::isfinite(f);
        const auto converged = track.converged(epsilon);
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }

        ++iteration;
    }

    return state;
}
