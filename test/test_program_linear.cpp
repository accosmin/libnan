#include <fixture/solver.h>
#include <function/program/cvx410.h>
#include <function/program/cvx48b.h>
#include <function/program/cvx48c.h>
#include <function/program/cvx48d.h>
#include <function/program/cvx48e.h>
#include <function/program/cvx48f.h>
#include <function/program/cvx49.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/linear.h>
#include <nano/function/quadratic.h>

using namespace nano;
using namespace constraint;

namespace
{
strings_t make_solver_ids()
{
    return {"ipm"}; // TODO: add penalty and augmented lagrangian
}

// TODO: test with duplicated and linearly dependant linear constraints
/*inline auto make_permutation(const tensor_size_t m)
{
    auto permutation = arange(0, m);
    std::shuffle(permutation.begin(), permutation.end(), make_rng());
    return permutation;
}

inline auto duplicate(const program::equality_t<matrix_t, vector_t>& equality, const scalar_t dep_w1,
                      const scalar_t dep_w2)
{
    const auto& A = equality.m_A;
    const auto& b = equality.m_b;

    const auto m = A.rows();
    const auto n = A.cols();

    auto b2 = vector_t{2 * m};
    auto A2 = matrix_t{2 * m, n};

    const auto permutation = make_permutation(m);
    for (tensor_size_t row = 0; row < m; ++row)
    {
        const auto permuted_row = permutation(row);
        const auto permuted_mix = (permuted_row + 1) % m;
        const auto duplicat_row = 2 * m - 1 - row;

        b2(row)          = b(permuted_row);
        b2(duplicat_row) = b(permuted_row) * dep_w1 + b(permuted_mix) * dep_w2;

        A2.row(row)          = A.row(permuted_row);
        A2.row(duplicat_row) = A.row(permuted_row).array() * dep_w1 + A.row(permuted_mix).array() * dep_w2;
    }

    return program::make_equality(A2, b2);
}

void check_solution(const function_t& function)
{
    // const auto [A, b, G, h] = make_linear_constraints(function);

    // TODO: test duplicated equality constraints
    if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 1.0, 0.0);

        check_with_logger([&](const logger_t& logger) { check_solution_penalty(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_program(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_augmented(dprogram, expected, logger); });
    }

    // TODO: test linearly dependant equality constraints
    if (program.m_eq.valid())
    {
        auto dprogram = program;
        dprogram.m_eq = duplicate(program.m_eq, 0.2, 1.1);

        check_with_logger([&](const logger_t& logger) { check_solution_penalty(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_program(dprogram, expected, logger); });
        check_with_logger([&](const logger_t& logger) { check_solution_augmented(dprogram, expected, logger); });
    }

    // test original program
    check_with_logger([&](const logger_t& logger) { check_solution_penalty(function, logger); });
    check_with_logger([&](const logger_t& logger) { check_solution_interior(function, logger); });
    check_with_logger([&](const logger_t& logger) { check_solution_augmented(function, logger); });
}*/
} // namespace

UTEST_BEGIN_MODULE(test_program_linear)

UTEST_CASE(constrain)
{
    const auto c = vector_t::zero(3);
    const auto a = vector_t::zero(3);
    const auto b = vector_t::zero(2);
    const auto A = matrix_t::zero(2, 3);

    auto function = linear_program_t{"lp", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(A * function.variable() >= b);
    UTEST_REQUIRE(A * function.variable() <= b);
    UTEST_REQUIRE(a * function.variable() == 1.0);
    UTEST_REQUIRE(a * function.variable() >= 1.0);
    UTEST_REQUIRE(a * function.variable() <= 1.0);
    UTEST_REQUIRE(function.variable() >= 1.0);
    UTEST_REQUIRE(function.variable() <= 1.0);
    UTEST_REQUIRE(!function.constrain(functional_equality_t{function}));
    UTEST_REQUIRE(!function.constrain(functional_inequality_t{function}));
    UTEST_REQUIRE(!function.constrain(euclidean_ball_equality_t{vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(euclidean_ball_inequality_t{vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(quadratic_equality_t{matrix_t::zero(3, 3), vector_t::zero(3), 0.0}));
    UTEST_REQUIRE(!function.constrain(quadratic_inequality_t{matrix_t::zero(3, 3), vector_t::zero(3), 0.0}));
}

UTEST_CASE(program1)
{
    // see example 13.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(-4, -2, 0, 0);
    const auto A = make_matrix<scalar_t>(2, 1, 1, 1, 0, 2, 0.5, 0, 1);
    const auto b = make_vector<scalar_t>(5, 8);
    const auto x = make_vector<scalar_t>(11.0 / 3.0, 4.0 / 3.0, 0.0, 0.0);

    auto function = linear_program_t{"lp1", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solver_ids(), function);
}

UTEST_CASE(program2)
{
    // see exercise 14.1, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto c = make_vector<scalar_t>(1, 0);
    const auto A = make_matrix<scalar_t>(1, 1, 1);
    const auto b = make_vector<scalar_t>(1);
    const auto x = make_vector<scalar_t>(0.0, 1.0);

    auto function = linear_program_t{"lp2", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(x));

    check_convexity(function);
    check_minimize(make_solver_ids(), function);
}

UTEST_CASE(program3)
{
    // NB: unbounded program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(1, 0, 1, 1);
    const auto b = make_vector<scalar_t>(2);

    auto function = linear_program_t{"lp3", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(optimum_t::status::unbounded));

    check_convexity(function);
    check_minimize(make_solver_ids(), function);
}

UTEST_CASE(program4)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0);
    const auto A = make_matrix<scalar_t>(2, 0, 1, 1, 0);
    const auto b = make_vector<scalar_t>(-1, -1);

    auto function = linear_program_t{"lp4", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(optimum_t::status::unfeasible));

    check_convexity(function);
    check_minimize(make_solver_ids(), function);
}

UTEST_CASE(program5)
{
    // NB: unfeasible program!
    const auto c = make_vector<scalar_t>(-1, 0, 0);
    const auto A = make_matrix<scalar_t>(3, 0, 1, 1, 0, 0, 1, 0, 1, 0);
    const auto b = make_vector<scalar_t>(1, 1, 1);

    auto function = linear_program_t{"lp5", c};
    UTEST_REQUIRE(A * function.variable() == b);
    UTEST_REQUIRE(function.variable() >= 0.0);
    UTEST_REQUIRE(function.optimum(optimum_t::status::unfeasible));

    check_convexity(function);
    check_minimize(make_solver_ids(), function);
}

UTEST_CASE(program_cvx48b)
{
    for (const tensor_size_t dims : {1, 7, 11})
    {
        for (const auto lambda : {-1.0, -1.42, -4.2, -42.1})
        {
            const auto function = linear_program_cvx48b_t{dims, lambda};

            check_convexity(function);
            check_minimize(make_solver_ids(), function);
        }
    }
}

UTEST_CASE(program_cvx48c)
{
    for (const tensor_size_t dims : {1, 7, 11})
    {
        const auto function = linear_program_cvx48c_t{dims};

        check_convexity(function);
        check_minimize(make_solver_ids(), function);
    }
}

UTEST_CASE(program_cvx48d_eq)
{
    for (const tensor_size_t dims : {2, 4, 9})
    {
        const auto function = linear_program_cvx48d_eq_t{dims};

        check_convexity(function);
        check_minimize(make_solver_ids(), function);
    }
}

UTEST_CASE(program_cvx48d_ineq)
{
    for (const tensor_size_t dims : {2, 5, 8})
    {
        const auto function = linear_program_cvx48d_ineq_t{dims};

        check_convexity(function);
        check_minimize(make_solver_ids(), function);
    }
}

UTEST_CASE(program_cvx48e_eq)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            const auto function = linear_program_cvx48e_eq_t{dims, alpha};

            check_convexity(function);
            check_minimize(make_solver_ids(), function);
        }
    }
}

UTEST_CASE(program_cvx48e_ineq)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (tensor_size_t alpha = 0; alpha <= dims; ++alpha)
        {
            const auto function = linear_program_cvx48e_ineq_t{dims, alpha};

            check_convexity(function);
            check_minimize(make_solver_ids(), function);
        }
    }
}

UTEST_CASE(program_cvx48f)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        for (const auto alpha : {0.0, 0.3, 0.7, 1.0})
        {
            const auto function = linear_program_cvx48f_t{dims, alpha};

            check_convexity(function);
            check_minimize(make_solver_ids(), function);
        }
    }
}

UTEST_CASE(program_cvx49)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        const auto function = linear_program_cvx49_t{dims};

        check_convexity(function);
        check_minimize(make_solver_ids(), function);
    }
}

UTEST_CASE(program_cvx410)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        const auto feasible = true;
        const auto function = linear_program_cvx410_t{dims, feasible};

        check_convexity(function);
        check_minimize(make_solver_ids(), function);
    }
}

UTEST_CASE(program_cvx410_unfeasible)
{
    for (const tensor_size_t dims : {2, 3, 5})
    {
        const auto feasible = false;
        const auto function = linear_program_cvx410_t{dims, feasible};

        check_convexity(function);
        check_minimize(make_solver_ids(), function);
    }
}

UTEST_END_MODULE()
