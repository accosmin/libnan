#include <fixture/function.h>
#include <fixture/loss.h>
#include <fixture/solver.h>
#include <fixture/tuner.h>
#include <nano/tuner/surrogate.h>
#include <nano/tuner/util.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
auto make_param_space1()
{
    return make_param_space("param1", param_space_t::type::linear, 0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
}

auto make_param_space2()
{
    return make_param_space("param2", param_space_t::type::log10, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1, 1e+2, 1e+3);
}

auto make_param_spaces()
{
    return param_spaces_t{make_param_space1(), make_param_space2()};
}

auto evaluateLL(const scalar_t x, const scalar_t y, const scalar_t x0, const scalar_t y0)
{
    return square(x - x0) + square(y - y0) + 0.5;
}

auto evaluate10(const scalar_t x, const scalar_t y, const scalar_t x0, const scalar_t y0)
{
    return square(x - x0) + square(std::log10(y) - std::log10(y0) - (1.0 + x) / (1.0 + x0) + 1.0) + 0.5;
}

template <class tevaluator>
void check_optimize(const tuner_t& tuner, const param_spaces_t& spaces, const tevaluator& evaluator)
{
    const auto  logger  = make_stdout_logger();
    const auto& params0 = spaces[0].values();
    const auto& params1 = spaces[1].values();

    for (tensor_size_t ix0 = 0; ix0 < params0.size(); ++ix0)
    {
        for (tensor_size_t iy0 = 0; iy0 < params1.size(); ++iy0)
        {
            const auto x0 = params0(ix0);
            const auto y0 = params1(iy0);

            const auto callback = [&](const tensor2d_t& params)
            {
                tensor1d_t values(params.size<0>());
                for (tensor_size_t itrial = 0; itrial < values.size(); ++itrial)
                {
                    values(itrial) = evaluator(params(itrial, 0), params(itrial, 1), x0, y0);
                }
                return values;
            };

            tuner_steps_t steps;
            UTEST_REQUIRE_NOTHROW(steps = tuner.optimize(spaces, callback, logger));
            UTEST_CHECK(std::is_sorted(steps.begin(), steps.end()));

            igrids_t igrids;
            for (const auto& step : steps)
            {
                UTEST_CHECK(std::find(igrids.begin(), igrids.end(), step.m_igrid) == igrids.end());
                igrids.push_back(step.m_igrid);
            }

            UTEST_CHECK_EQUAL(steps.begin()->m_igrid, make_indices(ix0, iy0));
            UTEST_CHECK_EQUAL(steps.begin()->m_param, make_tensor<scalar_t>(make_dims(2), x0, y0));
            UTEST_CHECK_EQUAL(steps.begin()->m_value, 0.5);
        }
    }
}

void check_minimizer(const function_t& function, const vector_t& optimum)
{
    const auto* const solver_id = function.smooth() ? "lbfgs" : "ellipsoid";

    const auto config  = minimize_config_t{};
    const auto solver  = make_solver(solver_id);
    solver->parameter("solver::max_evals") = 10000;
    const auto state   = check_minimize(*solver, function, make_random_x0(function), config);
    UTEST_CHECK_CLOSE(state.fx(), 0.0, 1e-6);
    UTEST_CHECK_CLOSE(state.x(), optimum, 1e-7);
}

void check_surrogate(const tensor1d_t& p, const tensor1d_t& q)
{
    const auto function = quadratic_surrogate_t(q.vector());
    check_gradient(function);
    check_convexity(function);
    check_minimizer(function, p.vector());
    UTEST_CHECK_EQUAL(function.size(), p.size());
}

void check_surrogate_fit(const tensor1d_t& q, const tensor2d_t& p, const tensor1d_t& y)
{
    for (const auto* const loss_id : {"mse", "mae"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss     = make_loss(loss_id);
        const auto function = quadratic_surrogate_fit_t(*loss, p, y);
        check_gradient(function);
        check_convexity(function);
        check_minimizer(function, q.vector());
        UTEST_CHECK_EQUAL(function.size(), q.size());
    }
}
} // namespace

UTEST_BEGIN_MODULE(test_tuner)

UTEST_CASE(factory)
{
    const auto& tuners = tuner_t::all();
    UTEST_CHECK_EQUAL(tuners.ids().size(), 2U);
    UTEST_CHECK(tuners.get("surrogate") != nullptr);
    UTEST_CHECK(tuners.get("local-search") != nullptr);
}

UTEST_CASE(param_space_empty)
{
    const auto make = [](const param_space_t::type type) { return make_param_space("param", type, tensor1d_t{}); };

    UTEST_CHECK_THROW(make(param_space_t::type::log10), std::runtime_error);
    UTEST_CHECK_THROW(make(param_space_t::type::linear), std::runtime_error);
}

UTEST_CASE(param_space_invalid)
{
    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::log10, -1.0, +1.0), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::log10, +1.0), std::runtime_error);
    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::linear, +1.0), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::log10, -1.0, +1.0, +1.0), std::runtime_error);
    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::linear, -1.0, +1.0, +1.0), std::runtime_error);

    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::log10, -1.0, +2.0, +1.0, +3.0),
                      std::runtime_error);
    UTEST_CHECK_THROW(make_param_space("param", param_space_t::type::linear, -1, 0, +2.0, +1.0, +3.0),
                      std::runtime_error);
}

UTEST_CASE(param_space_log10)
{
    const auto space = make_param_space("param", param_space_t::type::log10, 1e-6, 1e-3, 1e+1, 1e+2);

    UTEST_CHECK_CLOSE(space.to_surrogate(1e-5), -5.0, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(1e+0), +0.0, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(1e+2), +2.0, 1e-12);
    UTEST_CHECK_THROW(space.to_surrogate(3e-7), std::runtime_error);
    UTEST_CHECK_THROW(space.to_surrogate(1e+7), std::runtime_error);

    UTEST_CHECK_CLOSE(space.from_surrogate(-7.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(-6.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(-1.0), 1e-1, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+1.0), 1e+1, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+2.0), 1e+2, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+3.0), 1e+2, 1e-12);

    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-7.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-6.0), 1e-6, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-3.1), 1e-3, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.5), 1e+1, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+1.6), 1e+2, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+2.1), 1e+2, 1e-12);

    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(-7.0), 0);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(-6.0), 0);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(-3.1), 1);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.5), 2);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+1.6), 3);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+2.1), 3);
}

UTEST_CASE(param_space_linear)
{
    const auto type  = param_space_t::type::linear;
    const auto space = param_space_t{"param", type, make_tensor<scalar_t>(make_dims(4), 0.1, 0.2, 0.5, 1.0)};

    UTEST_CHECK_CLOSE(space.to_surrogate(0.10), +0.0, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(0.55), +0.5, 1e-12);
    UTEST_CHECK_CLOSE(space.to_surrogate(1.00), +1.0, 1e-12);
    UTEST_CHECK_THROW(space.to_surrogate(0.01), std::runtime_error);
    UTEST_CHECK_THROW(space.to_surrogate(1.01), std::runtime_error);

    UTEST_CHECK_CLOSE(space.from_surrogate(-1.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+0.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+0.5), 0.55, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+1.0), 1.00, 1e-12);
    UTEST_CHECK_CLOSE(space.from_surrogate(+2.0), 1.00, 1e-12);

    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(-1.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.0), 0.10, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.1), 0.20, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+0.5), 0.50, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+1.0), 1.00, 1e-12);
    UTEST_CHECK_CLOSE(space.closest_grid_value_from_surrogate(+1.1), 1.00, 1e-12);

    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(-1.0), 0);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.0), 0);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.1), 1);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.2), 1);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.3), 2);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.4), 2);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+0.5), 2);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+1.0), 3);
    UTEST_CHECK_EQUAL(space.closest_grid_point_from_surrogate(+1.1), 3);
}

UTEST_CASE(step)
{
    const auto lhs = tuner_step_t{make_indices(0, 0), make_tensor<scalar_t>(make_dims(2), 0.0, 0.0), 2.0};
    const auto rhs = tuner_step_t{make_indices(1, 1), make_tensor<scalar_t>(make_dims(2), 1.0, 1.0), 0.0};

    UTEST_CHECK(rhs < lhs);
}

UTEST_CASE(util)
{
    const auto spaces = make_param_spaces();
    const auto logger = make_null_logger();

    const auto min_igrid = make_min_igrid(spaces);
    const auto avg_igrid = make_avg_igrid(spaces);
    const auto max_igrid = make_max_igrid(spaces);

    UTEST_CHECK_EQUAL(min_igrid, make_indices(0, 0));
    UTEST_CHECK_EQUAL(max_igrid, make_indices(9, 6));
    UTEST_CHECK_EQUAL(avg_igrid, make_indices(5, 3));
    {
        const auto igrids = igrids_t{
            make_indices(0, 1),
            make_indices(9, 3),
            make_indices(6, 6),
        };
        const auto params = map_to_grid(spaces, igrids);
        UTEST_CHECK_EQUAL(params, make_tensor<scalar_t>(make_dims(3, 2), 0.0, 1e-2, 1.0, 1e+0, 0.7, 1e+3));
    }
    {
        const auto igrids = local_search(min_igrid, max_igrid, min_igrid, 1);
        UTEST_REQUIRE_EQUAL(igrids.size(), 4U);
        UTEST_CHECK_EQUAL(igrids[0], make_indices(0, 0));
        UTEST_CHECK_EQUAL(igrids[1], make_indices(0, 1));
        UTEST_CHECK_EQUAL(igrids[2], make_indices(1, 0));
        UTEST_CHECK_EQUAL(igrids[3], make_indices(1, 1));
    }
    {
        const auto igrids = local_search(min_igrid, max_igrid, max_igrid, 2);
        UTEST_REQUIRE_EQUAL(igrids.size(), 4U);
        UTEST_CHECK_EQUAL(igrids[0], make_indices(7, 4));
        UTEST_CHECK_EQUAL(igrids[1], make_indices(7, 6));
        UTEST_CHECK_EQUAL(igrids[2], make_indices(9, 4));
        UTEST_CHECK_EQUAL(igrids[3], make_indices(9, 6));
    }
    {
        const auto igrids = local_search(min_igrid, max_igrid, avg_igrid, 1);
        UTEST_REQUIRE_EQUAL(igrids.size(), 9U);
        UTEST_CHECK_EQUAL(igrids[0], make_indices(4, 2));
        UTEST_CHECK_EQUAL(igrids[1], make_indices(4, 3));
        UTEST_CHECK_EQUAL(igrids[2], make_indices(4, 4));
        UTEST_CHECK_EQUAL(igrids[3], make_indices(5, 2));
        UTEST_CHECK_EQUAL(igrids[4], make_indices(5, 3));
        UTEST_CHECK_EQUAL(igrids[5], make_indices(5, 4));
        UTEST_CHECK_EQUAL(igrids[6], make_indices(6, 2));
        UTEST_CHECK_EQUAL(igrids[7], make_indices(6, 3));
        UTEST_CHECK_EQUAL(igrids[8], make_indices(6, 4));
    }
    {
        const auto callback = [](const tensor2d_t& params)
        {
            static auto value = 0.0;

            tensor1d_t values(params.size<0>());
            for (tensor_size_t i = 0; i < values.size(); ++i)
            {
                values(i) = ++value;
            }
            return values;
        };

        tuner_steps_t steps;

        UTEST_CHECK(evaluate(spaces, callback, {min_igrid, max_igrid}, logger, steps));
        UTEST_REQUIRE_EQUAL(steps.size(), 2U);
        UTEST_CHECK_EQUAL(steps[0].m_igrid, min_igrid);
        UTEST_CHECK_EQUAL(steps[1].m_igrid, max_igrid);

        UTEST_CHECK(!evaluate(spaces, callback, {min_igrid, max_igrid}, logger, steps));
        UTEST_REQUIRE_EQUAL(steps.size(), 2U);
        UTEST_CHECK_EQUAL(steps[0].m_igrid, min_igrid);
        UTEST_CHECK_EQUAL(steps[1].m_igrid, max_igrid);

        UTEST_CHECK(evaluate(spaces, callback, {avg_igrid}, logger, steps));
        UTEST_REQUIRE_EQUAL(steps.size(), 3U);
        UTEST_CHECK_EQUAL(steps[0].m_igrid, min_igrid);
        UTEST_CHECK_EQUAL(steps[1].m_igrid, max_igrid);
        UTEST_CHECK_EQUAL(steps[2].m_igrid, avg_igrid);

        UTEST_CHECK(!evaluate(spaces, callback, {avg_igrid}, logger, steps));
        UTEST_REQUIRE_EQUAL(steps.size(), 3U);
        UTEST_CHECK_EQUAL(steps[0].m_igrid, min_igrid);
        UTEST_CHECK_EQUAL(steps[1].m_igrid, max_igrid);
        UTEST_CHECK_EQUAL(steps[2].m_igrid, avg_igrid);
    }
}

UTEST_CASE(quadratic_surrogate_1d)
{
    const auto p = make_tensor<scalar_t>(make_dims(1), +1.0);
    const auto q = make_tensor<scalar_t>(make_dims(3), +1.0, -2.0, +1.0);

    check_surrogate(p, q);
}

UTEST_CASE(quadratic_surrogate_2d)
{
    const auto p = make_tensor<scalar_t>(make_dims(2), +1.0, -2.0);
    const auto q = make_tensor<scalar_t>(make_dims(6), +5.0, -2.0, +4.0, +1.0, +0.0, +1.0);

    check_surrogate(p, q);
}

UTEST_CASE(quadratic_surrogate_2dc)
{
    const auto p = make_tensor<scalar_t>(make_dims(2), 0.1, 1.0);
    const auto q = make_tensor<scalar_t>(make_dims(6), 1.0, 0.0, -2.0, 1.0, -0.2, 1.01);

    check_surrogate(p, q);
}

UTEST_CASE(quadratic_surrogate_fit1d)
{
    const auto q = make_tensor<scalar_t>(make_dims(3), 1.0, 0.5, -1.0);
    const auto p = make_tensor<scalar_t>(make_dims(7, 1), -3.0, -2.0, -1.0, +0.0, +1.0, +2.0, +3.0);

    tensor1d_t y(7);
    for (tensor_size_t i = 0; i < y.size(); ++i)
    {
        const auto p0 = p(i, 0);
        y(i)          = q(0) * 1.0 + q(1) * p0 + q(2) * p0 * p0;
    }

    check_surrogate_fit(q, p, y);
}

UTEST_CASE(quadratic_surrogate_fit2d)
{
    const auto q = make_tensor<scalar_t>(make_dims(6), 1.0, 0.5, 1.5, 2.0, -1.0, -1.0);
    const auto p = make_tensor<scalar_t>(
        make_dims(25, 2), -2.0, -2.0, -2.0, -1.0, -2.0, +0.0, -2.0, +1.0, -2.0, +2.0, -1.0, -2.0, -1.0, -1.0, -1.0,
        +0.0, -1.0, +1.0, -1.0, +2.0, +0.0, -2.0, +0.0, -1.0, +0.0, +0.0, +0.0, +1.0, +0.0, +2.0, +1.0, -2.0, +1.0,
        -1.0, +1.0, +0.0, +1.0, +1.0, +1.0, +2.0, +2.0, -2.0, +2.0, -1.0, +2.0, +0.0, +2.0, +1.0, +2.0, +2.0);

    tensor1d_t y(25);
    for (tensor_size_t i = 0; i < y.size(); ++i)
    {
        const auto p0 = p(i, 0), p1 = p(i, 1);
        y(i) = q(0) * 1.0 + q(1) * p0 + q(2) * p1 + q(3) * p0 * p0 + q(4) * p0 * p1 + q(5) * p1 * p1;
    }

    check_surrogate_fit(q, p, y);
}

UTEST_CASE(local_search)
{
    const auto tuner = make_tuner("local-search");

    check_optimize(*tuner, make_param_spaces(), evaluateLL);
}

UTEST_CASE(surrogate)
{
    const auto tuner = make_tuner("surrogate");

    check_optimize(*tuner, make_param_spaces(), evaluate10);
}

UTEST_CASE(fails_empty_param_spaces)
{
    const auto spaces   = param_spaces_t{};
    const auto logger   = make_stdout_logger();
    const auto callback = [](const tensor2d_t&) { return tensor1d_t{}; };

    for (const auto& id : tuner_t::all().ids())
    {
        const auto tuner = make_tuner(id);
        UTEST_CHECK_THROW(tuner->optimize(spaces, callback, logger), std::runtime_error);
    }
}

UTEST_CASE(fails_invalid_param_values)
{
    const auto spaces   = make_param_spaces();
    const auto logger   = make_stdout_logger();
    const auto callback = [](const tensor2d_t& params)
    {
        const auto dims = make_dims(params.size<0>());
        return make_full_tensor<scalar_t>(dims, std::numeric_limits<scalar_t>::quiet_NaN());
    };

    for (const auto& id : tuner_t::all().ids())
    {
        const auto tuner = make_tuner(id);
        UTEST_CHECK_THROW(tuner->optimize(spaces, callback, logger), std::runtime_error);
    }
}

UTEST_END_MODULE()
