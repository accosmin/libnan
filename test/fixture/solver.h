#include <fixture/function.h>
#include <fixture/logger.h>
#include <nano/core/numeric.h>
#include <nano/solver.h>
#include <utest/utest.h>

using namespace nano;

[[maybe_unused]] inline auto make_solver(const string_t& name = "lbfgs")
{
    auto solver = solver_t::all().get(name);
    UTEST_REQUIRE(solver);
    return solver;
}

struct minimize_config_t
{
    auto& expected_minimum(const scalar_t value)
    {
        if (!std::isfinite(m_expected_minimum))
        {
            m_expected_minimum = value;
        }
        return *this;
    }

    auto& expected_maximum_deviation(const scalar_t value)
    {
        m_expected_maximum_deviation = value;
        return *this;
    }

    scalar_t m_expected_minimum{std::numeric_limits<scalar_t>::quiet_NaN()};
    scalar_t m_expected_maximum_deviation{1e-6};
};

struct solver_description_t
{
    solver_description_t() = default;

    auto& smooth_config(const minimize_config_t& config)
    {
        m_smooth_config = config;
        return *this;
    }

    auto& nonsmooth_config(const minimize_config_t& config)
    {
        m_nonsmooth_config = config;
        return *this;
    }

    minimize_config_t m_smooth_config{};
    minimize_config_t m_nonsmooth_config{};
};

[[maybe_unused]] static auto make_description(const string_t& solver_id)
{
    if (solver_id == "cgd-n" || solver_id == "cgd-hs" || solver_id == "cgd-fr" || solver_id == "cgd-pr" ||
        solver_id == "cgd-cd" || solver_id == "cgd-ls" || solver_id == "cgd-dy" || solver_id == "cgd-dycd" ||
        solver_id == "cgd-dyhs" || solver_id == "cgd-frpr" || solver_id == "lbfgs" || solver_id == "sr1" ||
        solver_id == "bfgs" || solver_id == "hoshino" || solver_id == "fletcher")
    {
        // NB: very fast, accurate and reliable on smooth problems.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-6))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-2));
    }
    else if (solver_id == "dfp")
    {
        // NB: DFP needs many more iterations to reach the solution for some smooth problems.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-5))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-2));
    }
    else if (solver_id == "gd")
    {
        // NB: gradient descent needs many more iterations to minimize badly conditioned problems.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-5))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-2));
    }
    else if (solver_id == "ellipsoid")
    {
        // NB: the ellipsoid method is reasonably fast only for very low-dimensional problems.
        // NB: the ellipsoid method is very precise (used as a reference) and very reliable.
        // NB: the stopping criterion is working very well in practice.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-6))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-6));
    }
    else if (solver_id == "rqb" || solver_id == "fpba1" || solver_id == "fpba2")
    {
        // NB: the (fast) proximal bundle algorithms are very precise and very reliable.
        // NB: the stopping criterion is working very well in practice.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-6))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-5));
    }
    else if (solver_id == "gs" || solver_id == "gs-lbfgs" || solver_id == "ags" || solver_id == "ags-lbfgs")
    {
        // NB: the gradient sampling methods are accurate for both smooth and non-smooth problems.
        // NB: the gradient sampling methods are very expensive on debug.
        // NB: the stopping criterion is working well in practice, but it needs many iterations.
        // NB: the adaptive gradient sampling methods are not very stable.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-5))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-4));
    }
    else if (solver_id == "sgm" || solver_id == "cocob" || solver_id == "sda" ||
             solver_id == "wda" ||                                             // primal-dual subgradient method
             solver_id == "pgm" || solver_id == "dgm" || solver_id == "fgm" || // universal gradient methods
             solver_id == "asga2" || solver_id == "asga4" ||                   // accelerated sub-gradient methods
             solver_id == "osga")                                              // optimal subgradient algorithm
    {
        // NB: unreliable methods:
        // - either no theoretical or practical stopping criterion
        // - very slow convergence rate for both non-smooth and hard smooth problems
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-3))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-1));
    }
    else if (solver_id == "ipm")
    {
        // NB: the interior point method can solve linear and quadratic convex programs very reliable.
        return solver_description_t{}
            .smooth_config(minimize_config_t{}.expected_maximum_deviation(1e-8))
            .nonsmooth_config(minimize_config_t{}.expected_maximum_deviation(1e-1));
    }
    else
    {
        assert(false);
        return solver_description_t{};
    }
}

[[maybe_unused]] static auto check_minimize(solver_t& solver, const function_t& function, const vector_t& x0,
                                            const minimize_config_t& config = minimize_config_t{})
{
    const auto op = [&](const logger_t& logger)
    {
        const auto state0      = solver_state_t{function, x0};
        const auto solver_id   = solver.type_id();
        const auto lsearch0_id = solver.type() == solver_type::line_search ? solver.lsearch0().type_id() : "N/A";
        const auto lsearchk_id = solver.type() == solver_type::line_search ? solver.lsearchk().type_id() : "N/A";

        logger.info(std::setprecision(10), function.name(), " ", solver_id, "[", lsearch0_id, ",", lsearchk_id,
                    "]\n:x0=[", state0.x().transpose(), "],", state0, "\n");

        function.clear_statistics();

        // minimize
        auto state = solver.minimize(function, x0, logger);
        UTEST_CHECK(state.valid());
        UTEST_CHECK_EQUAL(state.fcalls(), function.fcalls());
        UTEST_CHECK_EQUAL(state.gcalls(), function.gcalls());
        if (function.constraints().empty())
        {
            UTEST_CHECK_LESS_EQUAL(state.fx(), state0.fx() + epsilon1<scalar_t>());
        }

        const auto& optimum = function.optimum();
        // clang-format off
        UTEST_CHECK(optimum.m_xbest.size() == 0 ||
                    optimum.m_xbest.size() == state.x().size());
        // clang-format on

        // check optimum (if known and unique)
        if (optimum.m_xbest.size() == state.x().size())
        {
            UTEST_CHECK_CLOSE(state.x(), optimum.m_xbest, config.m_expected_maximum_deviation);
        }

        // check optimum function value (if known)
        if (std::isfinite(optimum.m_fbest))
        {
            UTEST_CHECK_CLOSE(state.fx(), optimum.m_fbest, config.m_expected_maximum_deviation);
        }
        if (function.convex() && std::isfinite(config.m_expected_minimum))
        {
            UTEST_CHECK_CLOSE(state.fx(), config.m_expected_minimum, config.m_expected_maximum_deviation);
        }

        // check convergence status
        switch (optimum.m_status)
        {
        case optimum_t::status::unfeasible:
            // unfeasible problem
            UTEST_CHECK_EQUAL(state.status(), solver_status::unfeasible);
            break;

        case optimum_t::status::unbounded:
            // unbounded problem
            UTEST_CHECK_EQUAL(state.status(), solver_status::unbounded);
            break;

        default:
            // solvable problem, check the expected convergence criterion if convergence reached
            switch (state.status())
            {
            case solver_status::value_test:
            {
                const auto epsilon  = solver.parameter("solver::epsilon").value<scalar_t>();
                const auto patience = solver.parameter("solver::patience").value<tensor_size_t>();
                UTEST_CHECK_LESS(state.value_test(patience), epsilon);
                break;
            }

            case solver_convergence::gradient_test:
            {
                const auto epsilon = solver.parameter("solver::epsilon").value<scalar_t>();
                UTEST_CHECK_LESS(state.gradient_test(), epsilon);
                break;
            }

            case solver_convergence::kkt_optimality_test:
            {
                const auto epsilon = solver.parameter("solver::epsilon").value<scalar_t>();
                UTEST_CHECK_LESS(state.feasibility_test(), epsilon);
                UTEST_CHECK_LESS(state.kkt_optimality_test(), epsilon);
                break;
            }

            case solver_convergence::specific_test:
                // NB: either no stopping criterion or a specific one, at least it shouldn't fail!
                UTEST_CHECK_NOT_EQUAL(state.status(), solver_status::failed);
                break;

            default:
                // NB: convergence not reached, expecting maximum iterations status without any failure!
                UTEST_CHECK_EQUAL(state.status(), solver_status::max_iters);
                break;
            }
        }

        return state;
    };
    return check_with_logger(op);
}

[[maybe_unused]] static void check_minimize(const rsolvers_t& solvers, const function_t& function)
{
    for (const auto& x0 : make_random_x0s(function))
    {
        auto expected_minimum = std::numeric_limits<scalar_t>::quiet_NaN();
        for (const auto& solver : solvers)
        {
            const auto& solver_id = solver->type_id();
            UTEST_NAMED_CASE(scat(function.name(), "/", solver_id));

            const auto descr = make_description(solver_id);

            auto config = function.smooth() ? descr.m_smooth_config : descr.m_nonsmooth_config;
            config.expected_minimum(expected_minimum);

            const auto state = check_minimize(*solver, function, x0, config);
            expected_minimum = state.fx();

            log_info(std::setprecision(10), function.name(), ": solver=", solver_id, ",fx=", state.fx(),
                     ",calls=", state.fcalls(), "|", state.gcalls(), ".\n");
        }
    }
}

[[maybe_unused]] static void check_minimize(const rsolvers_t& solvers, const rfunctions_t& functions)
{
    for (const auto& function : functions)
    {
        UTEST_REQUIRE(function);
        check_minimize(solvers, *function);
    }
}

[[maybe_unused]] static void check_minimize(const strings_t& solver_ids, const rfunctions_t& functions)
{
    auto solvers = rsolvers_t{};
    solvers.reserve(solver_ids.size());
    for (const auto& solver_id : solver_ids)
    {
        solvers.emplace_back(make_solver(solver_id));
    }

    check_minimize(solvers, functions);
}

[[maybe_unused]] static void check_minimize(const strings_t& solver_ids, const function_t& function)
{
    auto solvers = rsolvers_t{};
    solvers.reserve(solver_ids.size());
    for (const auto& solver_id : solver_ids)
    {
        solvers.emplace_back(make_solver(solver_id));
    }

    check_minimize(solvers, function);
}
