#include <solver/sgm.h>

using namespace nano;

solver_sgm_t::solver_sgm_t()
    : solver_t("sgm")
{
    register_parameter(parameter_t::make_scalar("solver::sgm::power", 0.5, LE, 0.75, LE, 1.0));
}

rsolver_t solver_sgm_t::clone() const
{
    return std::make_unique<solver_sgm_t>(*this);
}

solver_state_t solver_sgm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<int64_t>();
    const auto power     = parameter("solver::sgm::power").value<scalar_t>();

    auto state = solver_state_t{function, x0};
    auto x = state.x();
    auto g = state.gx();

    for (auto iteration = 0; function.fcalls() + function.gcalls() < max_evals; ++iteration)
    {
        if (g.lpNorm<Eigen::Infinity>() < std::numeric_limits<scalar_t>::epsilon())
        {
            solver_t::done_gradient_test(state, true, logger);
            break;
        }

        const auto lambda = 1.0 / std::pow(iteration + 1, power);
        x -= lambda * g / g.lpNorm<2>();

        const auto f = function(x, g);
        state.update_if_better(x, g, f);

        if (solver_t::done_value_test(state, true, logger))
        {
            break;
        }
    }

    return state;
}
