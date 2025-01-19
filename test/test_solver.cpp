#include <fixture/function.h>
#include <fixture/lsearch0.h>
#include <fixture/lsearchk.h>
#include <fixture/solver.h>
#include <function/benchmark/sphere.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>
#include <nano/solver.h>

using namespace nano;

namespace
{
template <class... tscalars>
void check_feasible(const function_t& function, const tscalars... point)
{
    auto state = solver_state_t{function, make_vector<scalar_t>(point...)};
    UTEST_CHECK(state.valid());
    UTEST_CHECK_LESS(state.feasibility_test(), epsilon0<scalar_t>());
}

template <class... tscalars>
void check_unfeasible(const function_t& function, const tscalars... point)
{
    auto state = solver_state_t{function, make_vector<scalar_t>(point...)};
    UTEST_CHECK(state.valid());
    UTEST_CHECK_GREATER(state.feasibility_test(), epsilon3<scalar_t>());
}
} // namespace

UTEST_BEGIN_MODULE(test_solver)

UTEST_CASE(state_str)
{
    for (const auto status : enum_values<solver_status>())
    {
        std::stringstream stream;
        stream << status;
        UTEST_CHECK_EQUAL(stream.str(), scat(status));
    }
}

UTEST_CASE(state_valid)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(state.valid());
}

UTEST_CASE(state_invalid_fNAN)
{
    const auto function = function_sphere_t{3};
    const auto state    = solver_state_t{function, make_vector<scalar_t>(NAN, 1.0, 0.0)};
    UTEST_CHECK(!state.valid());
}

UTEST_CASE(state_has_descent)
{
    const auto function = function_sphere_t{5};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(state.has_descent(-state.gx()));
}

UTEST_CASE(state_has_no_descent0)
{
    const auto function = function_sphere_t{2};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(!state.has_descent(make_vector<scalar_t>(.0, 0.0)));
}

UTEST_CASE(state_has_no_descent1)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_random_x0(function)};
    UTEST_CHECK(!state.has_descent(state.gx()));
}

UTEST_CASE(state_update_if_better)
{
    const auto function = function_sphere_t{2};
    const auto x0       = make_vector<scalar_t>(0.0, 0.0);
    const auto x1       = make_vector<scalar_t>(1.0, 1.0);
    const auto x2       = make_vector<scalar_t>(2.0, 2.0);
    const auto x09      = make_vector<scalar_t>(0.9, 0.9);

    // update path: x1, x2 (up), NaN (div), x1 (=), x09 (down), x0 (down)

    auto state = solver_state_t{function, x1};
    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(!state.update_if_better(x2, function(x2)));

    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(!state.update_if_better(x2, std::numeric_limits<scalar_t>::quiet_NaN()));

    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(!state.update_if_better(x1, function(x1)));

    UTEST_CHECK_CLOSE(state.x(), x1, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x1), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), std::numeric_limits<scalar_t>::max(), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), std::numeric_limits<scalar_t>::max(), 1e-12);

    UTEST_CHECK(state.update_if_better(x09, function(x09)));

    UTEST_CHECK_CLOSE(state.x(), x09, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x09), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), 0.38, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), 0.38, 1e-12);

    UTEST_CHECK(state.update_if_better(x0, function(x0)));

    UTEST_CHECK_CLOSE(state.x(), x0, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x0), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), 1.62, 1e-12);

    UTEST_CHECK(!state.update_if_better(x0, function(x0)));

    UTEST_CHECK_CLOSE(state.x(), x0, 1e-12);
    UTEST_CHECK_CLOSE(state.fx(), function(x0), 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(0), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(1), 0.0, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(2), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(3), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(4), 1.62, 1e-12);
    UTEST_CHECK_CLOSE(state.value_test(5), 1.62, 1e-12);
}

UTEST_CASE(state_convergence0)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_vector<scalar_t>(0, 0, 0, 0, 0, 0, 0)};
    UTEST_CHECK_GREATER_EQUAL(state.gradient_test(), 0);
    UTEST_CHECK_LESS(state.gradient_test(), epsilon0<scalar_t>());
}

UTEST_CASE(state_convergence1)
{
    const auto function = function_sphere_t{7};
    const auto state    = solver_state_t{function, make_random_x0(function, epsilon1<scalar_t>())};
    UTEST_CHECK_GREATER_EQUAL(state.gradient_test(), 0);
    UTEST_CHECK_LESS(state.gradient_test(), epsilon2<scalar_t>());
}

UTEST_CASE(state_convergence2)
{
    const auto function = function_sphere_t{3};
    const auto bstate   = solver_state_t{function, make_vector<scalar_t>(1e-3, 1e-3, 1e-3)};
    const auto cstate   = solver_state_t{function, make_vector<scalar_t>(1e-4, 1e-4, 1e-4)};
    UTEST_CHECK(::nano::converged(bstate, cstate, 1e-3));
    UTEST_CHECK(!::nano::converged(bstate, cstate, 1e-4));
}

UTEST_CASE(factory)
{
    for (const auto& solver_id : solver_t::all().ids())
    {
        UTEST_CHECK_NOTHROW(make_solver(solver_id));
        UTEST_CHECK_NOTHROW(make_description(solver_id));
    }
}

UTEST_CASE(config_solvers)
{
    for (const auto& solver_id : solver_t::all().ids())
    {
        const auto solver = make_solver(solver_id);

        // NB: 0 < c1 < c2 < 1
        UTEST_CHECK_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-4, 1e-1));
        UTEST_CHECK_EQUAL(solver->parameter("solver::tolerance").value_pair<scalar_t>(), std::make_tuple(1e-4, 1e-1));

        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(2e-1, 1e-1), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, 1e-4), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, +1.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, -0.1), std::runtime_error);
        UTEST_CHECK_THROW(solver->parameter("solver::tolerance") = std::make_tuple(-0.1, +1.1), std::runtime_error);
        UTEST_CHECK_EQUAL(solver->parameter("solver::tolerance").value_pair<scalar_t>(), std::make_tuple(1e-4, 1e-1));

        UTEST_CHECK_NOTHROW(solver->parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1));
        UTEST_CHECK_EQUAL(solver->parameter("solver::tolerance").value_pair<scalar_t>(), std::make_tuple(1e-1, 9e-1));

        UTEST_CHECK_THROW(solver->lsearch0("invalid-lsearch0-id"), std::runtime_error);
        UTEST_CHECK_THROW(solver->lsearchk("invalid-lsearchk-id"), std::runtime_error);

        UTEST_CHECK_NOTHROW(solver->lsearch0("constant"));
        UTEST_CHECK_NOTHROW(solver->lsearch0(*make_lsearch0("constant")));

        UTEST_CHECK_NOTHROW(solver->lsearchk("backtrack"));
        UTEST_CHECK_NOTHROW(solver->lsearchk(*make_lsearchk("backtrack")));
    }
}

UTEST_CASE(feasible_equality)
{
    {
        const auto A = make_matrix<scalar_t>(2, 2, 1, 0, 0, 1, 1);
        const auto b = make_vector<scalar_t>(3, 2);

        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(3)};
        UTEST_REQUIRE(A * function.variable() == b);

        check_feasible(function, 1.0, 1.0, 1.0);
        check_feasible(function, 1.5, 0.0, 2.0);
        check_unfeasible(function, 1.0, 1.0, 0.0);
        check_unfeasible(function, 0.0, 1.0, 1.0);
    }
}

UTEST_CASE(feasible_inequality)
{
    {
        const auto A = make_matrix<scalar_t>(2, 2, 1, 0, 0, 1, 1);
        const auto b = make_vector<scalar_t>(3, 2);

        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(3)};
        UTEST_REQUIRE(A * function.variable() <= b);

        check_feasible(function, 1.0, 1.0, 1.0);
        check_feasible(function, 1.5, 0.0, 2.0);
        check_feasible(function, 1.0, 1.0, 0.0);
        check_feasible(function, 0.0, 1.0, 1.0);
        check_unfeasible(function, 2.0, 1.0, 1.0);
        check_unfeasible(function, 1.0, 1.0, 2.0);
    }
    {
        const auto upper = make_vector<scalar_t>(+1, +1, +2);

        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(3)};
        UTEST_REQUIRE(function.variable() <= upper);

        check_feasible(function, -1.0, -1.0, 2.0);
        check_feasible(function, +0.0, +1.0, 1.0);
        check_feasible(function, +1.0, +1.0, 1.0);
        check_feasible(function, +1.0, +1.0, 2.0);
        check_unfeasible(function, 1.1, 1.0, 1.0);
        check_unfeasible(function, 1.0, 1.0, 2.1);
    }
    {
        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(3)};
        UTEST_REQUIRE(function.variable() <= 1.0);

        check_feasible(function, -1.0, -1.0, 1.0);
        check_feasible(function, +0.0, +1.0, 1.0);
        check_feasible(function, +1.0, +1.0, 1.0);
        check_unfeasible(function, 1.0, 1.0, 2.0);
        check_unfeasible(function, 1.1, 1.0, 1.0);
        check_unfeasible(function, 1.0, 1.0, 1.1);
    }
    {
        const auto lower = make_vector<scalar_t>(-1, -1, -1);

        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(3)};
        UTEST_REQUIRE(function.variable() >= lower);

        check_feasible(function, -1.0, -1.0, 2.0);
        check_feasible(function, +0.0, +1.0, 1.0);
        check_feasible(function, +1.0, +1.0, 1.0);
        check_unfeasible(function, -1.1, -1.0, -1.0);
        check_unfeasible(function, -1.0, -2.0, -3.0);
    }
    {
        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(3)};
        UTEST_REQUIRE(function.variable() >= 1.0);

        check_feasible(function, 1, 1, 1);
        check_feasible(function, 1, 2, 3);
        check_unfeasible(function, 0, 1, 1);
        check_unfeasible(function, 0, 0, 0);
    }
}

UTEST_CASE(feasible_convex_hull_center)
{
    for (tensor_size_t dims = 2; dims < 100; dims += 3)
    {
        auto function = linear_program_t{"lp", make_random_vector<scalar_t>(dims)};
        UTEST_REQUIRE(function.variable() <= 1.0);
        UTEST_REQUIRE(function.variable() >= 0.0);
        UTEST_REQUIRE(vector_t::constant(dims, 1.0) * function.variable() == 1.0);

        const auto x0 = vector_t{vector_t::constant(dims, 1.0 / static_cast<scalar_t>(dims))};

        auto state = solver_state_t{function, x0};
        UTEST_CHECK(state.valid());
        UTEST_CHECK_LESS(state.feasibility_test(), 5.0 * epsilon0<scalar_t>());
    }
}

UTEST_CASE(feasible_numopt131)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);

    check_feasible(function, 11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);
    check_feasible(function, 0.0, 4.0, 1.0, 6.0);
    check_feasible(function, 2.0, 2.0, 1.0, 3.0);
}

UTEST_CASE(feasible_numopt141)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);

    check_feasible(function, 0.0, 1.0);
    check_feasible(function, 1.0, 0.0);
    check_feasible(function, 0.1, 0.9);
}

UTEST_END_MODULE()
