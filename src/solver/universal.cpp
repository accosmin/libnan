#include <solver/universal.h>

using namespace nano;

solver_universal_t::solver_universal_t(string_t id)
    : solver_t(std::move(id))
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::universal::L0", 0.0, LT, 1e+0, LT, fmax));
    register_parameter(parameter_t::make_integer("solver::universal::lsearch_max_iters", 10, LE, 100, LE, 100));
}

solver_pgm_t::solver_pgm_t()
    : solver_universal_t("pgm")
{
}

rsolver_t solver_pgm_t::clone() const
{
    return std::make_unique<solver_pgm_t>(*this);
}

solver_state_t solver_pgm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonconvex(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto epsilon                = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals              = parameter("solver::max_evals").value<tensor_size_t>();
    const auto L0                     = parameter("solver::universal::L0").value<scalar_t>();
    const auto lsearch_max_iterations = parameter("solver::universal::lsearch_max_iters").value<tensor_size_t>();

    auto state = solver_state_t{function, x0};

    auto L    = L0;
    auto xk   = state.x();
    auto xk1  = state.x();
    auto gxk  = state.gx();
    auto gxk1 = state.gx();
    auto fxk  = state.fx();
    auto fxk1 = state.fx();

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // 1. line-search
        auto M         = L;
        auto iter_ok   = false;
        for (tensor_size_t k = 0; k < lsearch_max_iterations && !iter_ok && std::isfinite(fxk1); ++k)
        {
            xk1     = xk - gxk / M;
            fxk1    = function(xk1, gxk1);
            iter_ok = std::isfinite(fxk1) &&
                      fxk1 <= fxk + gxk.dot(xk1 - xk) + 0.5 * M * (xk1 - xk).dot(xk1 - xk) + 0.5 * epsilon;
            M *= 2.0;
        }

        if (iter_ok)
        {
            // 2. update current and best state (if the line-search step doesn't fail)
            L   = 0.5 * M;
            xk  = xk1;
            gxk = gxk1;
            fxk = fxk1;
            state.update_if_better(xk1, gxk1, fxk1);
        }

        if (solver_t::done_value_test(state, iter_ok, logger))
        {
            break;
        }
    }

    return state;
}

solver_dgm_t::solver_dgm_t()
    : solver_universal_t("dgm")
{
}

rsolver_t solver_dgm_t::clone() const
{
    return std::make_unique<solver_dgm_t>(*this);
}

solver_state_t solver_dgm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonconvex(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto epsilon                = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals              = parameter("solver::max_evals").value<tensor_size_t>();
    const auto L0                     = parameter("solver::universal::L0").value<scalar_t>();
    const auto lsearch_max_iterations = parameter("solver::universal::lsearch_max_iters").value<tensor_size_t>();

    auto state = solver_state_t{function, x0};

    auto L    = L0;
    auto yk   = state.x();
    auto xk1  = state.x();
    auto gxk  = state.gx();
    auto gxk1 = state.gx();
    auto gphi = x0;
    auto fxk1 = state.fx();

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // 1. line-search
        auto M         = L;
        auto iter_ok   = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok && std::isfinite(fxk1); ++k)
        {
            xk1  = gphi - gxk / M;
            fxk1 = function(xk1, gxk1);
            iter_ok =
                std::isfinite(fxk1) && function(yk = xk1 - gxk1 / M) <= fxk1 - 0.5 * gxk1.dot(gxk1) / M + 0.5 * epsilon;
            M *= 2.0;
        }

        if (iter_ok)
        {
            gphi -= gxk / M;

            // 2. update current and best state (if the line-search step doesn't fail)
            L   = 0.5 * M;
            gxk = gxk1;
            state.update_if_better(xk1, gxk1, fxk1);
        }

        if (solver_t::done_value_test(state, iter_ok, logger))
        {
            break;
        }
    }

    return state;
}

solver_fgm_t::solver_fgm_t()
    : solver_universal_t("fgm")
{
}

rsolver_t solver_fgm_t::clone() const
{
    return std::make_unique<solver_fgm_t>(*this);
}

solver_state_t solver_fgm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonconvex(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto epsilon                = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals              = parameter("solver::max_evals").value<tensor_size_t>();
    const auto L0                     = parameter("solver::universal::L0").value<scalar_t>();
    const auto lsearch_max_iterations = parameter("solver::universal::lsearch_max_iters").value<tensor_size_t>();

    auto state = solver_state_t{function, x0};

    auto L    = L0;
    auto Ak   = 0.0;
    auto ak1  = 0.0;
    auto vk   = x0;
    auto yk   = x0;
    auto yk1  = x0;
    auto xk1  = x0;
    auto gxk1 = state.gx();
    auto gyk1 = state.gx();
    auto fxk1 = state.fx();
    auto fyk1 = state.fx();

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // 2. line-search
        auto M         = L;
        auto iter_ok   = false;
        for (int64_t k = 0; k < lsearch_max_iterations && !iter_ok && std::isfinite(fxk1) && std::isfinite(fyk1); ++k)
        {
            ak1            = (1.0 + std::sqrt(1.0 + 4.0 * M * Ak)) / (2.0 * M);
            const auto tau = ak1 / (Ak + ak1);

            xk1  = tau * vk + (1.0 - tau) * yk;
            fxk1 = function(xk1, gxk1);

            yk1  = tau * (vk - ak1 * gxk1) + (1.0 - tau) * yk;
            fyk1 = function(yk1, gyk1);

            iter_ok = std::isfinite(fxk1) && std::isfinite(fyk1) &&
                      fyk1 <= fxk1 + gxk1.dot(yk1 - xk1) + 0.5 * M * (yk1 - xk1).dot(yk1 - xk1) + 0.5 * epsilon * tau;
            M *= 2.0;
        }

        if (iter_ok)
        {
            // 3. update current and best state (if the line-search step doesn't fail)
            yk = yk1;
            Ak += ak1;
            L = 0.5 * M;
            vk -= ak1 * gxk1;
            state.update_if_better(yk1, gyk1, fyk1);
        }

        if (solver_t::done_value_test(state, iter_ok, logger))
        {
            break;
        }
    }

    return state;
}
