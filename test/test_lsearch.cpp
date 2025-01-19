#include <fixture/function.h>
#include <fixture/lsearchk.h>
#include <iomanip>

using namespace nano;

namespace
{
auto make_functions()
{
    return function_t::make({1, 16, convexity::ignore, smoothness::yes, 10}, std::regex(".+"));
}

void test(const rlsearchk_t& lsearch, const function_t& function, const vector_t& x0, const scalar_t t0,
          const std::tuple<scalar_t, scalar_t>& c12)
{
    UTEST_REQUIRE_NOTHROW(lsearch->parameter("lsearchk::tolerance") = c12);

    const auto [c1, c2]       = c12;
    const auto old_n_failures = utest_n_failures.load();
    const auto state0         = solver_state_t{function, x0};
    const auto descent        = vector_t{-state0.gx()};

    UTEST_CHECK(state0.valid());
    if (state0.gx().lpNorm<Eigen::Infinity>() < std::numeric_limits<scalar_t>::epsilon())
    {
        return;
    }
    UTEST_CHECK(state0.has_descent(descent));

    std::stringstream stream;
    stream << std::setprecision(10) << function.name() << " " << lsearch->type_id() << ": x0=["
           << state0.x().transpose() << "],t0=" << t0 << ",f0=" << state0.fx() << ",g0=" << state0.gradient_test()
           << "\n";

    const auto logger = make_stream_logger(stream);

    const auto cgdescent_epsilon = [&]()
    { return lsearch->parameter("lsearchk::cgdescent::epsilon").value<scalar_t>(); };

    // check the Armijo and the Wolfe-like conditions are valid after line-search
    auto state                 = state0;
    const auto [ok, step_size] = lsearch->get(state, descent, t0, logger);
    UTEST_CHECK(ok);
    UTEST_CHECK(state.valid());
    UTEST_CHECK_GREATER(step_size, 0.0);
    UTEST_CHECK_LESS_EQUAL(state.fx(), state0.fx());

    switch (lsearch->type())
    {
    case lsearch_type::armijo: UTEST_CHECK(state.has_armijo(state0, descent, step_size, c1)); break;

    case lsearch_type::wolfe:
        UTEST_CHECK(state.has_armijo(state0, descent, step_size, c1));
        UTEST_CHECK(state.has_wolfe(state0, descent, c2));
        break;

    case lsearch_type::strong_wolfe:
        UTEST_CHECK(state.has_armijo(state0, descent, step_size, c1));
        UTEST_CHECK(state.has_strong_wolfe(state0, descent, c2));
        break;

    case lsearch_type::wolfe_approx_wolfe:
        UTEST_CHECK((state.has_armijo(state0, descent, step_size, c1) && state.has_wolfe(state0, descent, c2)) ||
                    (state.has_approx_armijo(state0, cgdescent_epsilon() * std::fabs(state0.fx())) &&
                     state.has_approx_wolfe(state0, descent, c1, c2)));
        break;

    default: break;
    }

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str();
    }
}

void test(const rlsearchk_t& lsearch, const function_t& function)
{
    for (const auto& x0 : make_random_x0s(function))
    {
        for (const auto& c12 : std::vector<std::tuple<scalar_t, scalar_t>>{
                 {1e-4, 1e-1},
                 {1e-4, 9e-1},
                 {1e-1, 9e-1}
        })
        {
            test(lsearch, function, x0, 1e-1, c12);
            test(lsearch, function, x0, 3e-1, c12);
            test(lsearch, function, x0, 1e+0, c12);
            test(lsearch, function, x0, 3e+1, c12);
        }
    }
}
} // namespace

UTEST_BEGIN_MODULE(test_lsearch)

UTEST_CASE(lsearch_type_str)
{
    UTEST_CHECK_EQUAL(scat(lsearch_type::none), "N/A");
    UTEST_CHECK_EQUAL(scat(lsearch_type::armijo), "Armijo");
    UTEST_CHECK_EQUAL(scat(lsearch_type::wolfe), "Wolfe");
    UTEST_CHECK_EQUAL(scat(lsearch_type::strong_wolfe), "strong Wolfe");
    UTEST_CHECK_EQUAL(scat(lsearch_type::wolfe_approx_wolfe), "Wolfe or approximative Wolfe");
}

UTEST_CASE(interpolate)
{
    const auto u = lsearch_step_t{4.2, 1.0, 0.5};
    const auto v = u;

    const auto tc = lsearch_step_t::interpolate(u, v, interpolation_type::cubic);
    const auto tq = lsearch_step_t::interpolate(u, v, interpolation_type::quadratic);
    const auto tb = lsearch_step_t::interpolate(u, v, interpolation_type::bisection);

    UTEST_CHECK_CLOSE(tc, 4.2, 1e-16);
    UTEST_CHECK_CLOSE(tq, 4.2, 1e-16);
    UTEST_CHECK_CLOSE(tb, 4.2, 1e-16);
}

UTEST_CASE(backtrack_cubic)
{
    const auto lsearch                                       = make_lsearchk("backtrack");
    lsearch->parameter("lsearchk::backtrack::interpolation") = interpolation_type::cubic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(backtrack_quadratic)
{
    const auto lsearch                                       = make_lsearchk("backtrack");
    lsearch->parameter("lsearchk::backtrack::interpolation") = interpolation_type::quadratic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(backtrack_bisection)
{
    const auto lsearch                                       = make_lsearchk("backtrack");
    lsearch->parameter("lsearchk::backtrack::interpolation") = interpolation_type::bisection;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(lemarechal_cubic)
{
    const auto lsearch                                        = make_lsearchk("lemarechal");
    lsearch->parameter("lsearchk::lemarechal::interpolation") = interpolation_type::cubic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(lemarechal_quadratic)
{
    const auto lsearch                                        = make_lsearchk("lemarechal");
    lsearch->parameter("lsearchk::lemarechal::interpolation") = interpolation_type::quadratic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(lemarechal_bisection)
{
    const auto lsearch                                        = make_lsearchk("lemarechal");
    lsearch->parameter("lsearchk::lemarechal::interpolation") = interpolation_type::bisection;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(morethuente)
{
    const auto lsearch = make_lsearchk("morethuente");

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(fletcher_cubic)
{
    const auto lsearch                                      = make_lsearchk("fletcher");
    lsearch->parameter("lsearchk::fletcher::interpolation") = interpolation_type::cubic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(fletcher_quadratic)
{
    const auto lsearch                                      = make_lsearchk("fletcher");
    lsearch->parameter("lsearchk::fletcher::interpolation") = interpolation_type::quadratic;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(fletcher_bisection)
{
    const auto lsearch                                      = make_lsearchk("fletcher");
    lsearch->parameter("lsearchk::fletcher::interpolation") = interpolation_type::bisection;

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_CASE(cgdescent)
{
    const auto lsearch = make_lsearchk("cgdescent");

    for (const auto& function : make_functions())
    {
        test(lsearch, *function);
    }
}

UTEST_END_MODULE()
