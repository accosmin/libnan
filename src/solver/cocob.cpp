#include <solver/cocob.h>

using namespace nano;

solver_cocob_t::solver_cocob_t()
    : solver_t("cocob")
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::cocob::L0-smooth", 0.0, LT, 1e-16, LE, fmax));
    register_parameter(parameter_t::make_scalar("solver::cocob::L0-nonsmooth", 0.0, LT, 1e+3, LE, fmax));
}

rsolver_t solver_cocob_t::clone() const
{
    return std::make_unique<solver_cocob_t>(*this);
}

solver_state_t solver_cocob_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonconvex(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto max_evals    = parameter("solver::max_evals").value<int>();
    const auto L0_smooth    = parameter("solver::cocob::L0-smooth").value<scalar_t>();
    const auto L0_nonsmooth = parameter("solver::cocob::L0-nonsmooth").value<scalar_t>();
    const auto L0           = function.smooth() ? L0_smooth : L0_nonsmooth;

    auto state = solver_state_t{function, x0}; // NB: keeps track of the best state

    auto x  = state.x();
    auto gx = state.gx();

    auto L      = make_full_vector<scalar_t>(x0.size(), L0);
    auto G      = make_full_vector<scalar_t>(x0.size(), 0.0);
    auto theta  = make_full_vector<scalar_t>(x0.size(), 0.0);
    auto reward = make_full_vector<scalar_t>(x0.size(), 0.0);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // compute parameter update
        L.array() = L.array().max(gx.array().abs());
        G.array() += gx.array().abs();

        theta -= gx.vector();
        reward = (reward.array() - (x - x0).array() * gx.array()).max(0.0);

        const auto beta = (theta.array() / (G + L).array()).tanh() / L.array();
        x               = x0.array() + beta * (L + reward).array();

        const auto fx = function(x, gx);
        state.update_if_better(x, gx, fx);

        const auto iter_ok   = std::isfinite(fx);
        if (solver_t::done_value_test(state, iter_ok, logger))
        {
            break;
        }
    }

    return state;
}
