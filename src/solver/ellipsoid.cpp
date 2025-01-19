#include <solver/ellipsoid.h>

using namespace nano;

solver_ellipsoid_t::solver_ellipsoid_t()
    : solver_t("ellipsoid")
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::ellipsoid::R", 0.0, LT, 1e+1, LT, fmax));
}

rsolver_t solver_ellipsoid_t::clone() const
{
    return std::make_unique<solver_ellipsoid_t>(*this);
}

solver_state_t solver_ellipsoid_t::do_minimize(const function_t& function, const vector_t& x0,
                                               const logger_t& logger) const
{
    solver_t::warn_nonconvex(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto R         = parameter("solver::ellipsoid::R").value<scalar_t>();
    const auto epsilon   = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals = parameter("solver::max_evals").value<int64_t>();

    auto state = solver_state_t{function, x0}; // best state

    auto x = state.x();
    auto f = state.fx();
    auto g = state.gx();

    const auto n = static_cast<scalar_t>(function.size());

    auto H = matrix_t{matrix_t::identity(function.size(), function.size())};
    H.array() *= function.size() == 1 ? R : (R * R);

    auto xv = x.vector();
    auto gv = g.vector();
    auto Hm = H.matrix();

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        const auto gHg = gv.dot(Hm * gv);
        if (gHg < std::numeric_limits<scalar_t>::epsilon())
        {
            solver_t::done_specific_test(state, true, true, logger);
            break;
        }

        if (function.size() == 1)
        {
            // NB: the ellipsoid method becomes bisection for the 1D case.
            x.array() += H(0) * (g(0) < 0.0 ? +1.0 : -1.0);
            Hm /= 2.0;
        }
        else
        {
            // NB: deep-cut variation
            const auto alpha = (f - state.fx()) / std::sqrt(gHg);

            xv.noalias() = xv - (1 + n * alpha) / (n + 1) * (Hm * gv) / std::sqrt(gHg);
            Hm.noalias() = (n * n) / (n * n - 1) * (1 - alpha * alpha) *
                           (Hm - 2 * (1 + n * alpha) / (n + 1) / (1 + alpha) * (Hm * gv * gv.transpose() * Hm) / gHg);
        }

        f = function(x, g);
        state.update_if_better(x, g, f);

        const auto iter_ok   = std::isfinite(f);
        const auto converged = std::sqrt(gHg) < epsilon;
        if (solver_t::done_specific_test(state, iter_ok, converged, logger))
        {
            break;
        }
    }

    return state;
}
