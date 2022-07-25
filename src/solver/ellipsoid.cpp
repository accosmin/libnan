#include <nano/solver/ellipsoid.h>

using namespace nano;

solver_ellipsoid_t::solver_ellipsoid_t()
{
    monotonic(false);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::ellipsoid::R", 0.0, LT, 1e+1, LT, fmax));
}

solver_state_t solver_ellipsoid_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto R         = parameter("solver::ellipsoid::R").value<scalar_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();

    auto state = solver_state_t{function, x0};

    auto f = state.f;
    auto x = state.x, g = state.g;

    const auto n = static_cast<scalar_t>(function.size());

    matrix_t H = matrix_t::Identity(function.size(), function.size());
    H.array() *= R * R;

    for (int64_t i = 0; function.fcalls() < max_evals; ++i)
    {
        const auto gHg = g.dot(H * g);
        if (gHg < std::numeric_limits<scalar_t>::epsilon())
        {
            const auto iter_ok   = true;
            const auto converged = true;
            solver_t::done(function, state, iter_ok, converged);
            break;
        }

        if (function.size() == 1)
        {
            // NB: the ellipsoid method becomes bisection for the 1D case.
            x.array() += H(0, 0) * (g(0) < 0.0 ? +1.0 : -1.0);
            H /= 2.0;
        }
        else
        {
            // NB: deep-cut variation
            const auto alpha = (f - state.f) / std::sqrt(gHg);

            x.noalias() = x - (1 + n * alpha) / (n + 1) * (H * g) / std::sqrt(gHg);
            H.noalias() = (n * n) / (n * n - 1) * (1 - alpha * alpha) *
                          (H - 2 * (1 + n * alpha) / (n + 1) / (1 + alpha) * (H * g * g.transpose() * H) / gHg);
        }

        f = function.vgrad(x, &g);
        state.update_if_better(x, g, f);

        const auto iter_ok   = std::isfinite(f);
        const auto converged = std::sqrt(gHg) < epsilon;
        if (solver_t::done(function, state, iter_ok, converged))
        {
            break;
        }
    }

    return state;
}