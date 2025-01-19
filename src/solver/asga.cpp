#include <solver/asga.h>

using namespace nano;

namespace
{
auto solve_sk1(const scalar_t miu, const scalar_t Sk, const scalar_t Lk1)
{
    const auto r = 1.0 + Sk * miu;
    return (r + std::sqrt(r * r + 4.0 * Lk1 * Sk * r)) / (2.0 * Lk1);
}

auto lsearch_done(const vector_t& y, scalar_t fy, const vector_t& x, const scalar_t fx, const vector_t& gx,
                  const scalar_t Lk, const scalar_t alphak, const scalar_t epsilon)
{
    return fy <= fx + gx.dot(y - x) + 0.5 * Lk * (y - x).squaredNorm() + 0.5 * alphak * epsilon;
}
} // namespace

solver_asga_t::solver_asga_t(string_t id)
    : solver_t(std::move(id))
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::asga::L0", 0.0, LT, 1e+0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::asga::gamma1", 1.0, LT, 4.0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::asga::gamma2", 0.0, LT, 0.9, LT, 1.0));
    register_parameter(parameter_t::make_integer("solver::asga::lsearch_max_iters", 10, LE, 100, LE, 1000));
}

solver_asga2_t::solver_asga2_t()
    : solver_asga_t("asga2")
{
}

rsolver_t solver_asga2_t::clone() const
{
    return std::make_unique<solver_asga2_t>(*this);
}

solver_state_t solver_asga2_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto epsilon           = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals         = parameter("solver::max_evals").value<tensor_size_t>();
    const auto L0                = parameter("solver::asga::L0").value<scalar_t>();
    const auto gamma1            = parameter("solver::asga::gamma1").value<scalar_t>();
    const auto gamma2            = parameter("solver::asga::gamma2").value<scalar_t>();
    const auto lsearch_max_iters = parameter("solver::asga::lsearch_max_iters").value<int>();
    const auto miu               = function.strong_convexity();

    auto state = solver_state_t{function, x0};
    if (done_gradient_test(state, true, logger))
    {
        return state;
    }

    auto Lk  = L0;
    auto Sk  = 0.0;
    auto fxk = std::numeric_limits<scalar_t>::max();

    auto xk        = x0;
    auto xk1       = x0;
    auto gxk1      = vector_t{x0.size()};
    auto zk        = x0;
    auto zk1       = x0;
    auto yk        = vector_t{x0.size()};
    auto gyk       = vector_t{x0.size()};
    auto sum_skgyk = make_full_vector<scalar_t>(x0.size(), 0.0);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        auto sk1     = 0.0;
        auto Lk1     = Lk / gamma1;
        auto Sk1     = Sk;
        auto fxk1    = fxk;
        auto iter_ok = false;
        for (auto p = 0; p < lsearch_max_iters && !iter_ok; ++p)
        {
            Lk1 *= gamma1;
            sk1 = solve_sk1(miu, Sk, Lk1);
            Sk1 = Sk + sk1;

            const auto alphak = sk1 / Sk1;
            yk                = alphak * zk + (1.0 - alphak) * xk;
            const auto fyk    = function(yk, gyk);

            zk1  = (x0 + sum_skgyk + sk1 * (miu * yk - gyk)) / (1.0 + miu * Sk1);
            xk1  = alphak * zk1 + (1.0 - alphak) * xk;
            fxk1 = function(xk1, gxk1);

            iter_ok = std::isfinite(Lk1) && std::isfinite(fxk1) && std::isfinite(fyk) &&
                      lsearch_done(xk1, fxk1, yk, fyk, gyk, Lk1, alphak, epsilon);
        }

        state.update_if_better(xk1, gxk1, fxk1);

        if (done_value_test(state, iter_ok, logger))
        {
            break;
        }

        xk  = xk1;
        zk  = zk1;
        Sk  = Sk1;
        fxk = fxk1;
        Lk  = gamma2 * Lk1;
        sum_skgyk += sk1 * (miu * yk - gyk);
    }

    return state;
}

solver_asga4_t::solver_asga4_t()
    : solver_asga_t("asga4")
{
}

rsolver_t solver_asga4_t::clone() const
{
    return std::make_unique<solver_asga4_t>(*this);
}

solver_state_t solver_asga4_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    warn_nonconvex(function, logger);
    warn_constrained(function, logger);

    const auto epsilon           = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals         = parameter("solver::max_evals").value<tensor_size_t>();
    const auto L0                = parameter("solver::asga::L0").value<scalar_t>();
    const auto gamma1            = parameter("solver::asga::gamma1").value<scalar_t>();
    const auto gamma2            = parameter("solver::asga::gamma2").value<scalar_t>();
    const auto lsearch_max_iters = parameter("solver::asga::lsearch_max_iters").value<int>();
    const auto miu               = function.strong_convexity();

    auto state = solver_state_t{function, x0};
    if (done_gradient_test(state, true, logger))
    {
        return state;
    }

    auto Lk  = L0;
    auto Sk  = 0.0;
    auto fyk = std::numeric_limits<scalar_t>::max();

    auto vk       = x0;
    auto yk       = x0;
    auto xk1      = vector_t{x0.size()};
    auto yk1      = vector_t{x0.size()};
    auto uk1      = vector_t{x0.size()};
    auto gxk1     = vector_t{x0.size()};
    auto gyk1     = vector_t{x0.size()};
    auto sum_skgk = make_full_vector<scalar_t>(x0.size(), 0.0);

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        auto sk1     = 0.0;
        auto Sk1     = Sk;
        auto Lk1     = Lk / gamma1;
        auto fyk1    = fyk;
        auto iter_ok = false;
        for (auto p = 0; p < lsearch_max_iters && !iter_ok; ++p)
        {
            Lk1 *= gamma1;
            sk1 = solve_sk1(miu, Sk, Lk1);
            Sk1 = Sk + sk1;

            const auto alphak = sk1 / Sk1;
            xk1               = alphak * vk + (1.0 - alphak) * yk;
            const auto fxk1   = function(xk1, gxk1);

            uk1  = (vk + sk1 * (miu * xk1 - gxk1)) / (1.0 + miu * sk1);
            yk1  = alphak * uk1 + (1.0 - alphak) * yk;
            fyk1 = function(yk1, gyk1);

            iter_ok = std::isfinite(Lk1) && std::isfinite(fxk1) && std::isfinite(fyk1) &&
                      lsearch_done(yk1, fyk1, xk1, fxk1, gxk1, Lk1, alphak, epsilon);
        }

        state.update_if_better(yk1, gyk1, fyk1);

        if (done_value_test(state, iter_ok, logger))
        {
            break;
        }

        yk  = yk1;
        Sk  = Sk1;
        fyk = fyk1;
        Lk  = gamma2 * Lk1;

        sum_skgk += sk1 * (miu * xk1 - gxk1);
        vk = (x0 + sum_skgk) / (1.0 + miu * Sk);
    }

    return state;
}
