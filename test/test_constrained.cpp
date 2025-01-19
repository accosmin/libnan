#include <fixture/function.h>
#include <fixture/solver.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/function/lambda.h>
#include <nano/function/penalty.h>
#include <nano/solver/augmented.h>
#include <nano/solver/penalty.h>

using namespace nano;
using namespace nano::constraint;

namespace
{
void check_penalty(penalty_function_t& penalty_function, const convexity expected_convexity,
                   const smoothness expected_smoothness, const scalar_t expected_strong_convexity)
{
    for (const auto penalty : {1e-1, 1e+0, 1e+1, 1e+2, 1e+3})
    {
        const auto trials  = 100;
        const auto epsilon = 1e-7;

        penalty_function.penalty(penalty);

        check_convexity(penalty_function);
        check_gradient(penalty_function, trials, epsilon);
        UTEST_CHECK_EQUAL(penalty_function.strong_convexity(), expected_strong_convexity);
        UTEST_CHECK_EQUAL(penalty_function.convex(), expected_convexity == convexity::yes);
        UTEST_CHECK_EQUAL(penalty_function.smooth(), expected_smoothness == smoothness::yes);
    }
}

template <class tpenalty>
void check_penalty(const function_t& function, const convexity expected_convexity, const smoothness expected_smoothness,
                   const scalar_t expected_strong_convexity)
{
    if constexpr (std::is_same_v<tpenalty, augmented_lagrangian_function_t>)
    {
        const auto n_equalities   = function.n_equalities();
        const auto n_inequalities = function.n_inequalities();

        const vector_t lambda = make_random_tensor<scalar_t>(make_dims(n_equalities), -1.0, +1.0).vector();
        const vector_t miu    = make_random_tensor<scalar_t>(make_dims(n_inequalities), +0.0, +1.0).vector();

        auto penalty_function = tpenalty{function, lambda, miu};
        check_penalty(penalty_function, expected_convexity, expected_smoothness, expected_strong_convexity);
    }
    else
    {
        auto penalty_function = tpenalty{function};
        check_penalty(penalty_function, expected_convexity, expected_smoothness, expected_strong_convexity);
    }
}

void check_penalties(const function_t& function, const convexity expected_convexity,
                     const smoothness expected_smoothness, const scalar_t expected_strong_convexity)
{
    const auto unconstrained = function.constraints().empty();

    check_penalty<linear_penalty_function_t>(
        function, expected_convexity, unconstrained ? expected_smoothness : smoothness::no, expected_strong_convexity);

    check_penalty<quadratic_penalty_function_t>(function, expected_convexity, expected_smoothness,
                                                expected_strong_convexity);

    check_penalty<augmented_lagrangian_function_t>(function, expected_convexity, expected_smoothness,
                                                   expected_strong_convexity);
}

template <class tpenalty>
void check_penalty(const function_t& function, const vector_t& x, const bool expected_valid)
{
    UTEST_CHECK_EQUAL(function.valid(x), expected_valid);

    auto penalty_function = tpenalty{function};

    for (const auto penalty : {1e-1, 1e+0, 1e+1, 1e+2, 1e+3})
    {
        const auto fx = function(x);
        const auto qx = penalty_function.penalty(penalty)(x);
        if (expected_valid)
        {
            UTEST_CHECK_CLOSE(fx, qx, 1e-16);
        }
        else
        {
            UTEST_CHECK_LESS(fx + 1e-8, qx);
        }
    }
}

void check_penalties(const function_t& function, const vector_t& x, const bool expected_valid)
{
    check_penalty<linear_penalty_function_t>(function, x, expected_valid);
    check_penalty<quadratic_penalty_function_t>(function, x, expected_valid);
}

void check_minimize(solver_t& solver, const function_t& function, const vector_t& x0, const vector_t& xbest,
                    const scalar_t fbest, const scalar_t epsilon)
{
    std::stringstream stream;
    stream << std::setprecision(10) << function.name() << "\n:x0=[" << x0.transpose() << "]\n";

    const auto logger = make_stream_logger(stream);

    function.clear_statistics();
    const auto state = solver.minimize(function, x0, logger);

    const auto old_n_failures = utest_n_failures.load();

    UTEST_CHECK(state.valid());
    UTEST_CHECK_CLOSE(state.x(), xbest, epsilon);
    UTEST_CHECK_CLOSE(state.fx(), fbest, epsilon);
    UTEST_CHECK_LESS_EQUAL(0.0, state.kkt_optimality_test1());
    UTEST_CHECK_LESS_EQUAL(0.0, state.kkt_optimality_test2());
    UTEST_CHECK_LESS(state.kkt_optimality_test1(), epsilon);
    UTEST_CHECK_LESS(state.kkt_optimality_test2(), epsilon);
    UTEST_CHECK_EQUAL(state.status(), solver_status::converged);
    UTEST_CHECK_EQUAL(state.fcalls(), function.fcalls());
    UTEST_CHECK_EQUAL(state.gcalls(), function.gcalls());

    if (old_n_failures != utest_n_failures.load())
    {
        std::cout << stream.str() << "\n";
    }
}

void check_penalty_solver(const function_t& function, const vector_t& xbest, const scalar_t fbest,
                          const scalar_t epsilon_nonsmooth, const scalar_t epsilon_smooth = 1e-6)
{
    if (linear_penalty_function_t{function}.convex())
    // NB: cannot solve non-convex non-smooth problems precisely!
    {
        UTEST_NAMED_CASE(scat(function.name(), "_linear_penalty_solver"));

        auto solver = solver_linear_penalty_t{};
        for (const auto& x0 : make_random_x0s(function, 5.0))
        {
            check_minimize(*solver.clone(), function, x0, xbest, fbest, epsilon_nonsmooth);
        }
    }
    {
        UTEST_NAMED_CASE(scat(function.name(), "_quadratic_penalty_solver"));

        auto solver = solver_quadratic_penalty_t{};
        for (const auto& x0 : make_random_x0s(function, 5.0))
        {
            check_minimize(*solver.clone(), function, x0, xbest, fbest, epsilon_smooth);
        }
    }
    {
        UTEST_NAMED_CASE(scat(function.name(), "_augmented_lagrangian_solver"));

        auto solver = solver_augmented_lagrangian_t{};
        for (const auto& x0 : make_random_x0s(function, 5.0))
        {
            check_minimize(*solver.clone(), function, x0, xbest, fbest, epsilon_smooth);
        }
    }
}

class sum_function_t final : public function_t
{
public:
    explicit sum_function_t(tensor_size_t size)
        : function_t("sum", size)
    {
        convex(convexity::yes);
        smooth(smoothness::yes);
    }

    rfunction_t clone() const override { return std::make_unique<sum_function_t>(*this); }

    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override
    {
        if (gx.size() == x.size())
        {
            gx.array() = 1.0;
        }
        return x.sum();
    }
};

class cauchy_function_t final : public function_t
{
public:
    explicit cauchy_function_t(tensor_size_t size)
        : function_t("cauchy", size)
    {
        convex(convexity::no);
        smooth(smoothness::yes);
    }

    rfunction_t clone() const override { return std::make_unique<cauchy_function_t>(*this); }

    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override
    {
        if (gx.size() == x.size())
        {
            gx = 2.0 * x / (0.36 + x.dot(x));
        }
        return std::log(0.36 + x.dot(x));
    }
};

class sumabsm1_function_t final : public function_t
{
public:
    explicit sumabsm1_function_t(tensor_size_t size)
        : function_t("sumabsm1", size)
    {
        convex(convexity::yes);
        smooth(smoothness::no);
    }

    rfunction_t clone() const override { return std::make_unique<sumabsm1_function_t>(*this); }

    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override
    {
        if (gx.size() == x.size())
        {
            gx.array() = x.array().sign();
        }
        return x.array().abs().sum() - 1.0;
    }
};
} // namespace

UTEST_BEGIN_MODULE(test_constrained)

UTEST_CASE(minimum)
{
    const auto constraint = constraint_t{
        minimum_t{1.0, 0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(::nano::is_linear(constraint));
    UTEST_CHECK(!::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.9, 1.0)), 0.1, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(2.0, 0.0)), 0.0, 1e-15);
}

UTEST_CASE(maximum)
{
    const auto constraint = constraint_t{
        maximum_t{1.0, 1}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(::nano::is_linear(constraint));
    UTEST_CHECK(!::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.9, 0.9)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(2.0, 1.2)), 0.2, 1e-15);
}

UTEST_CASE(constant)
{
    const auto constraint = constraint_t{
        constant_t{1.0, 1}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(::nano::is_linear(constraint));
    UTEST_CHECK(::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.9, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.1)), 0.1, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(2.0, 0.8)), 0.2, 1e-15);
}

UTEST_CASE(euclidean_ball_equality)
{
    const auto constraint = constraint_t{
        euclidean_ball_equality_t{make_vector<scalar_t>(0.0, 0.0), 1.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(!::nano::is_linear(constraint));
    UTEST_CHECK(::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 2.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 2.0)), 4.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0)), 1.0, 1e-15);
}

UTEST_CASE(euclidean_ball_inequality)
{
    const auto constraint = constraint_t{
        euclidean_ball_inequality_t{make_vector<scalar_t>(0.0, 0.0), 1.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(!::nano::is_linear(constraint));
    UTEST_CHECK(!::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 2.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 2.0)), 4.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0)), 0.0, 1e-15);
}

UTEST_CASE(linear_equality)
{
    const auto constraint = constraint_t{
        linear_equality_t{make_vector<scalar_t>(1.0, 1.0), -2.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(::nano::is_linear(constraint));
    UTEST_CHECK(::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 2.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(2.0, 2.0)), 2.0, 1e-15);
}

UTEST_CASE(linear_inequality)
{
    const auto constraint = constraint_t{
        linear_inequality_t{make_vector<scalar_t>(1.0, 1.0), -2.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(::nano::is_linear(constraint));
    UTEST_CHECK(!::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 2.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(2.0, 2.0)), 2.0, 1e-15);
}

UTEST_CASE(quadratic_equality)
{
    const auto constraint = constraint_t{
        quadratic_equality_t{make_matrix<scalar_t>(2, 1.0, 2.0, 2.0, 1.0), make_vector<scalar_t>(1.0, 1.0), -5.0}
    };

    UTEST_CHECK(!::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(!::nano::is_linear(constraint));
    UTEST_CHECK(::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0)), 5.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0)), 3.5, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0)), 3.5, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 2.0)), 4.5, 1e-15);
}

UTEST_CASE(quadratic_inequality)
{
    const auto constraint = constraint_t{
        quadratic_inequality_t{make_matrix<scalar_t>(3, 2.0, -1., 0.0, -1., 2.0, -1., 0.0, -1., 2.0),
                               make_vector<scalar_t>(1.0, 1.0, 1.0), -2.0}
    };

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(::nano::smooth(constraint));
    UTEST_CHECK(!::nano::is_linear(constraint));
    UTEST_CHECK(!::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 2.0 - std::sqrt(2.0), 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0, 1.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 1.0, 1.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 0.0, 1.0)), 2.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0, 1.0)), 2.0, 1e-15);
}

UTEST_CASE(functional_equality)
{
    const auto constraint = constraint_t{functional_equality_t{std::make_unique<sumabsm1_function_t>(3)}};

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(!::nano::smooth(constraint));
    UTEST_CHECK(!::nano::is_linear(constraint));
    UTEST_CHECK(::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.9, 0.9, 0.0)), 0.8, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.0, 0.0)), 1.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(2.0, 1.2, 0.0)), 2.2, 1e-15);
}

UTEST_CASE(functional_inequality)
{
    const auto constraint = constraint_t{functional_inequality_t{std::make_unique<sumabsm1_function_t>(3)}};

    UTEST_CHECK(::nano::convex(constraint));
    UTEST_CHECK(!::nano::smooth(constraint));
    UTEST_CHECK(!::nano::is_linear(constraint));
    UTEST_CHECK(!::nano::is_equality(constraint));
    UTEST_CHECK_CLOSE(::nano::strong_convexity(constraint), 0.0, 1e-15);

    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.0, 0.0, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.5, 0.2, 0.0)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.5, 0.0, -0.5)), 0.0, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(1.0, 1.2, 0.0)), 1.2, 1e-15);
    UTEST_CHECK_CLOSE(::nano::valid(constraint, make_vector<scalar_t>(0.1, -0.7, -0.4)), 0.2, 1e-15);
}

UTEST_CASE(noconstraint_convex_smooth)
{
    auto function = sum_function_t{3};
    UTEST_CHECK_EQUAL(function.constraints().size(), 0U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(function, make_random_x0(function), true);
    }
}

UTEST_CASE(noconstraint_convex_nonsmooth)
{
    auto function = sumabsm1_function_t{3};
    UTEST_CHECK_EQUAL(function.constraints().size(), 0U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::yes, smoothness::no, 0.0);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(function, make_random_x0(function), true);
    }
}

UTEST_CASE(noconstraint_nonconvex_smooth)
{
    auto function = cauchy_function_t{3};
    UTEST_CHECK_EQUAL(function.constraints().size(), 0U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::no, smoothness::yes, 0.0);
    for (auto trial = 0; trial < 100; ++trial)
    {
        check_penalties(function, make_random_x0(function), true);
    }
}

UTEST_CASE(constrained_box_one)
{
    auto function = sum_function_t{3};
    UTEST_CHECK(!(function.variable(-1) >= -0.5));
    UTEST_CHECK(!(function.variable(-1) <= +0.5));
    UTEST_CHECK(!(function.variable(+3) >= -0.5));
    UTEST_CHECK(!(function.variable(+3) <= +0.5));
    UTEST_CHECK(function.variable(+2) >= -0.5);
    UTEST_CHECK(function.variable(+2) <= +0.5);
    UTEST_CHECK_EQUAL(function.constraints().size(), 2U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 2);

    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    check_penalties(function, make_vector<scalar_t>(-0.1, -0.1, -0.1), true);
    check_penalties(function, make_vector<scalar_t>(+0.2, +0.2, +0.2), true);
    check_penalties(function, make_vector<scalar_t>(+0.5, +0.5, +0.5), true);
    check_penalties(function, make_vector<scalar_t>(-0.7, -0.7, -0.7), false);
    check_penalties(function, make_vector<scalar_t>(+0.8, +0.8, +0.8), false);
    check_penalties(function, make_vector<scalar_t>(-0.7, +0.1, +0.0), true);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.2, -0.7), false);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.6, +0.0), true);
}

UTEST_CASE(constrained_box_all)
{
    auto function = sum_function_t{3};
    UTEST_CHECK(function.variable() >= -0.5);
    UTEST_CHECK(function.variable() <= +0.5);
    UTEST_CHECK_EQUAL(function.constraints().size(), 6U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 6);

    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.1, +0.0), true);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.1, +0.4), true);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.6, +0.0), false);
    check_penalties(function, make_vector<scalar_t>(-0.2, -0.3, +1.0), false);
}

UTEST_CASE(constrained_box_vector)
{
    auto function = sum_function_t{3};
    UTEST_CHECK(!(function.variable() >= make_vector<scalar_t>(-0.5, -0.5, -0.5, -0.5)));
    UTEST_CHECK(!(function.variable() <= make_vector<scalar_t>(+0.5, +0.5, +0.5, +0.5)));
    UTEST_CHECK(function.variable() >= make_vector<scalar_t>(-0.5, -0.5, -0.5));
    UTEST_CHECK(function.variable() <= make_vector<scalar_t>(+0.5, +0.5, +0.5));
    UTEST_CHECK_EQUAL(function.constraints().size(), 6U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 6);

    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.1, +0.0), true);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.1, +0.4), true);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.6, +0.0), false);
    check_penalties(function, make_vector<scalar_t>(-0.2, -0.3, +1.0), false);
}

UTEST_CASE(constrained_constant)
{
    auto function = sumabsm1_function_t{3};
    UTEST_CHECK(!function.constrain(constant_t{1.0, -1}));
    UTEST_CHECK(!function.constrain(constant_t{1.0, +3}));
    UTEST_CHECK(function.constrain(constant_t{1.0, 2}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 1);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::yes, smoothness::no, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.5, 1.5, 1.0), true);
    check_penalties(function, make_vector<scalar_t>(1.0, 1.0, 1.0), true);
    check_penalties(function, make_vector<scalar_t>(0.1, 0.2, 0.3), false);
    check_penalties(function, make_vector<scalar_t>(0.1, 1.2, 1.3), false);
    check_penalties(function, make_vector<scalar_t>(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_euclidean_ball_inequality)
{
    auto function = sum_function_t{3};
    UTEST_CHECK(!function.constrain(euclidean_ball_inequality_t{make_vector<scalar_t>(1.0, 1.0, 1.0, 1.0), 1.0}));
    UTEST_CHECK(!function.constrain(euclidean_ball_inequality_t{make_vector<scalar_t>(1.0, 1.0, 1.0), 0.0}));
    UTEST_CHECK(function.constrain(euclidean_ball_inequality_t{make_vector<scalar_t>(0.0, 0.0, 0.0), 1.0}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 1);

    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.0, 0.0), true);
    check_penalties(function, make_vector<scalar_t>(0.5, 0.5, 0.5), true);
    check_penalties(function, make_vector<scalar_t>(0.6, 0.6, 0.6), false);
    check_penalties(function, make_vector<scalar_t>(1.0, 1.0, 1.0), false);
}

UTEST_CASE(constrained_affine_equality)
{
    auto function = sumabsm1_function_t{3};
    UTEST_CHECK(!function.constrain(linear_equality_t{make_vector<scalar_t>(1.0, 1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK(function.constrain(linear_equality_t{make_vector<scalar_t>(1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 1);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::yes, smoothness::no, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.5, 1.5, 1.0), true);
    check_penalties(function, make_vector<scalar_t>(1.0, 1.0, 1.0), true);
    check_penalties(function, make_vector<scalar_t>(0.1, 0.2, 0.3), false);
    check_penalties(function, make_vector<scalar_t>(0.1, 1.2, 1.3), false);
    check_penalties(function, make_vector<scalar_t>(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_affine_inequality)
{
    auto function = sumabsm1_function_t{3};
    UTEST_CHECK(!function.constrain(linear_inequality_t{make_vector<scalar_t>(1.0, 1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK(function.constrain(linear_inequality_t{make_vector<scalar_t>(1.0, 1.0, 1.0), -3.0}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 1);

    check_penalties(function, convexity::yes, smoothness::no, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.1, 0.2, 0.3), true);
    check_penalties(function, make_vector<scalar_t>(0.1, 1.2, 1.3), true);
    check_penalties(function, make_vector<scalar_t>(0.5, 1.5, 2.5), false);
}

UTEST_CASE(constrained_cauchy_inequality)
{
    auto function = cauchy_function_t{3};
    UTEST_CHECK(!function.constrain(functional_inequality_t{std::make_unique<cauchy_function_t>(4)}));
    UTEST_CHECK(function.constrain(functional_inequality_t{std::make_unique<cauchy_function_t>(3)}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 1);

    check_penalties(function, convexity::no, smoothness::yes, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.0, 0.0), true);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.0, 0.7), true);
    check_penalties(function, make_vector<scalar_t>(0.8, 0.0, 0.0), true);
    check_penalties(function, make_vector<scalar_t>(0.1, 0.2, 0.3), true);
    check_penalties(function, make_vector<scalar_t>(0.8, 0.1, 0.0), false);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.9, 0.0), false);
}

UTEST_CASE(constrained_sumabsm1_equality)
{
    auto function = sum_function_t{3};
    UTEST_CHECK(!function.constrain(functional_equality_t{std::make_unique<sumabsm1_function_t>(4)}));
    UTEST_CHECK(function.constrain(functional_equality_t{std::make_unique<sumabsm1_function_t>(3)}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 1);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::no, smoothness::no, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.0, 1.0), true);
    check_penalties(function, make_vector<scalar_t>(-0.9, 0.1, 0.0), true);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.9, 0.0), false);
    check_penalties(function, make_vector<scalar_t>(-0.6, +0.8, 0.1), false);
    check_penalties(function, make_vector<scalar_t>(-1.6, +0.8, 0.1), false);
}

UTEST_CASE(constrained_sumabsm1_inequality)
{
    auto function = sum_function_t{3};
    UTEST_CHECK(!function.constrain(functional_inequality_t{std::make_unique<sumabsm1_function_t>(4)}));
    UTEST_CHECK(function.constrain(functional_inequality_t{std::make_unique<sumabsm1_function_t>(3)}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 1);

    check_penalties(function, convexity::yes, smoothness::no, 0.0);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.0, 1.0), true);
    check_penalties(function, make_vector<scalar_t>(0.0, 0.9, 0.0), true);
    check_penalties(function, make_vector<scalar_t>(-0.6, +0.2, 0.1), true);
    check_penalties(function, make_vector<scalar_t>(-1.6, +0.8, 0.1), false);
    check_penalties(function, make_vector<scalar_t>(-0.2, +0.8, 0.1), false);
}

UTEST_CASE(constrained_quadratic2x2_inequality)
{
    auto q2 = make_vector<scalar_t>(1.0, 1.0);
    auto q3 = make_vector<scalar_t>(1.0, 1.0, 1.0);

    auto P2x2 = make_matrix<scalar_t>(2, 1.0, 2.0, 2.0, 1.0);
    auto P2x3 = make_matrix<scalar_t>(2, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0);
    auto P3x2 = make_matrix<scalar_t>(3, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0);

    auto function = sum_function_t{2};
    UTEST_CHECK(!function.constrain(quadratic_inequality_t{P2x2, q3, 1.0}));
    UTEST_CHECK(!function.constrain(quadratic_inequality_t{P2x3, q2, 1.0}));
    UTEST_CHECK(!function.constrain(quadratic_inequality_t{P3x2, q2, 1.0}));
    UTEST_CHECK(function.constrain(quadratic_inequality_t{P2x2, q2, 1.0}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 1);

    check_penalties(function, convexity::no, smoothness::yes, 0.0);
}

UTEST_CASE(constrained_quadratic3x3_inequality)
{
    auto q3 = make_vector<scalar_t>(1.0, 1.0, 1.0);
    auto q4 = make_vector<scalar_t>(1.0, 1.0, 1.0, 1.0);

    auto P3x3 = make_matrix<scalar_t>(3, 2.0, -1., 0.0, -1., 2.0, -1., 0.0, -1., 2.0);
    auto P3x4 = make_matrix<scalar_t>(3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    auto P4x3 = make_matrix<scalar_t>(4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    auto function = sum_function_t{3};
    UTEST_CHECK(!function.constrain(quadratic_inequality_t{P3x3, q4, 1.0}));
    UTEST_CHECK(!function.constrain(quadratic_inequality_t{P3x4, q3, 1.0}));
    UTEST_CHECK(!function.constrain(quadratic_inequality_t{P4x3, q3, 1.0}));
    UTEST_CHECK(function.constrain(quadratic_inequality_t{P3x3, q3, 1.0}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 0);
    UTEST_CHECK_EQUAL(n_inequalities(function), 1);

    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
}

UTEST_CASE(constrained_quadratic3x3_equality)
{
    auto q3 = make_vector<scalar_t>(1.0, 1.0, 1.0);
    auto q4 = make_vector<scalar_t>(1.0, 1.0, 1.0, 1.0);

    auto P3x3 = make_matrix<scalar_t>(3, 2.0, -1., 0.0, -1., 2.0, -1., 0.0, -1., 2.0);
    auto P3x4 = make_matrix<scalar_t>(3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    auto P4x3 = make_matrix<scalar_t>(4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    auto function = sum_function_t{3};
    UTEST_CHECK(!function.constrain(quadratic_equality_t{P3x3, q4, 1.0}));
    UTEST_CHECK(!function.constrain(quadratic_equality_t{P3x4, q3, 1.0}));
    UTEST_CHECK(!function.constrain(quadratic_equality_t{P4x3, q3, 1.0}));
    UTEST_CHECK(function.constrain(quadratic_equality_t{P3x3, q3, 1.0}));
    UTEST_CHECK_EQUAL(function.constraints().size(), 1U);
    UTEST_CHECK_EQUAL(n_equalities(function), 1);
    UTEST_CHECK_EQUAL(n_inequalities(function), 0);

    check_penalties(function, convexity::no, smoothness::yes, 0.0);
}

UTEST_CASE(minimize_objective1)
{
    // see 17.3, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        if (gx.size() == x.size())
        {
            gx.array() = 1.0;
        }
        return x.sum();
    };
    auto function = make_function(2, convexity::yes, smoothness::yes, 0.0, lambda);
    UTEST_CHECK(function.constrain(euclidean_ball_equality_t{make_vector<scalar_t>(0.0, 0.0), std::sqrt(2.0)}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::no, smoothness::yes, 0.0);
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 0.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(-2.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 2.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 1.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(-1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 1.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(-1.0, 0.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(-1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 1.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(-1.0, 1.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(0.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    const auto fbest = -2.0;
    const auto xbest = make_vector<scalar_t>(-1.0, -1.0);
    check_penalty_solver(function, xbest, fbest, 1e-4);
}

UTEST_CASE(minimize_objective2)
{
    // see 17.5, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        if (gx.size() == x.size())
        {
            gx(0) = -10.0 * x(0);
            gx(1) = +2.0 * x(1);
        }
        return -5.0 * x(0) * x(0) + x(1) * x(1);
    };
    auto function = make_function(2, convexity::no, smoothness::yes, 0.0, lambda);
    UTEST_CHECK(function.constrain(constant_t{1.0, 0}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::no, smoothness::yes, 0.0);
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 0.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(-1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 1.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 3.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(-1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 1.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(1.0, 3.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(0.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    const auto fbest = -5.0;
    const auto xbest = make_vector<scalar_t>(1.0, 0.0);
    check_penalty_solver(function, xbest, fbest, 1e-4);
}

UTEST_CASE(minimize_objective3)
{
    // see 17.24, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        if (gx.size() == x.size())
        {
            gx.array() = 1.0;
        }
        return x.sum();
    };
    auto function = make_function(1, convexity::yes, smoothness::yes, 0.0, lambda);
    UTEST_CHECK(function.constrain(minimum_t{1.0, 0}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 1.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(1.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(0.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(2.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(-1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    const auto fbest = +1.0;
    const auto xbest = make_vector<scalar_t>(1.0);
    check_penalty_solver(function, xbest, fbest, 1e-4);
}

UTEST_CASE(minimize_objective4)
{
    // see 15.34, "Numerical optimization", Nocedal & Wright, 2nd edition
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        if (gx.size() == x.size())
        {
            gx(0) = 4.0 * x(0) - 1.0;
            gx(1) = 4.0 * x(1);
        }
        return 2.0 * (x(0) * x(0) + x(1) * x(1) - 1.0) - x(0);
    };
    auto function = make_function(2, convexity::yes, smoothness::yes, 4.0, lambda);
    UTEST_CHECK(function.constrain(euclidean_ball_equality_t{make_vector<scalar_t>(0.0, 0.0), 1.0}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::no, smoothness::yes, 4.0);
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 0.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(-1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 1.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 1.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(0.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(1.0, 1.0)};
        UTEST_CHECK_CLOSE(state.ceq(), make_vector<scalar_t>(1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 1.0, 1e-12);
    }
    const auto fbest = -1.0;
    const auto xbest = make_vector<scalar_t>(1.0, 0.0);
    check_penalty_solver(function, xbest, fbest, 1e-4);
}

UTEST_CASE(minimize_objective5)
{
    // see 12.36, "Numerical optimization", Nocedal & Wright, 2nd edition
    // NB: the convention for the inequality constraints in the library is the opposite!
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        if (gx.size() == x.size())
        {
            gx(0) = 2.0 * (x(0) - 1.5);
            gx(1) = 4.0 * cube(x(1) - 0.5);
        }
        return square(x(0) - 1.5) + quartic(x(1) - 0.5);
    };
    auto function = make_function(2, convexity::yes, smoothness::yes, 0.0, lambda);
    UTEST_CHECK(function.constrain(linear_inequality_t{make_vector<scalar_t>(-1.0, -1.0), -1.0}));
    UTEST_CHECK(function.constrain(linear_inequality_t{make_vector<scalar_t>(-1.0, +1.0), -1.0}));
    UTEST_CHECK(function.constrain(linear_inequality_t{make_vector<scalar_t>(+1.0, -1.0), -1.0}));
    UTEST_CHECK(function.constrain(linear_inequality_t{make_vector<scalar_t>(+1.0, +1.0), -1.0}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 0.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(-1.0, -1.0, -1.0, -1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 1.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(-2.0, 0.0, -2.0, 0.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(1.0, 1.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(-3.0, -1.0, -1.0, 1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 1.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    const auto fbest = 5.0 / 16.0;
    const auto xbest = make_vector<scalar_t>(1.0, 0.0);
    check_penalty_solver(function, xbest, fbest, 1e-2);
}

UTEST_CASE(minimize_objective6)
{
    // see 12.56, "Numerical optimization", Nocedal & Wright, 2nd edition
    // NB: the convention for the inequality constraints in the library is the opposite!
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        if (gx.size() == x.size())
        {
            gx(0) = 1.0;
            gx(1) = 0.0;
        }
        return x(0);
    };
    auto function = make_function(2, convexity::yes, smoothness::yes, 0.0, lambda);
    UTEST_CHECK(function.constrain(minimum_t{0.0, 1}));
    UTEST_CHECK(function.constrain(euclidean_ball_inequality_t{make_vector<scalar_t>(1.0, 0.0), 1.0}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::yes, smoothness::yes, 0.0);
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(1.0, 0.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(0.0, -1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 0.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(0.0, 1.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(-1.0, 1.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 1.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }
    {
        const auto state = solver_state_t{function, make_vector<scalar_t>(-1.0, -1.0)};
        UTEST_CHECK_CLOSE(state.cineq(), make_vector<scalar_t>(1.0, 4.0), 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test1(), 4.0, 1e-12);
        UTEST_CHECK_CLOSE(state.kkt_optimality_test2(), 0.0, 1e-12);
    }

    const auto fbest = 0.0;
    const auto xbest = make_vector<scalar_t>(0.0, 0.0);
    check_penalty_solver(function, xbest, fbest, 1e-1);
}

UTEST_CASE(minimize_objective7)
{
    // see exercise 4.3, "Convex optimization", Boyd & Vanderberghe
    const auto lambda = [](vector_cmap_t x, vector_map_t gx)
    {
        static const auto P = make_matrix<scalar_t>(3, 13, 12, -2, 12, 17, 6, -2, 6, 12);
        static const auto q = make_vector<scalar_t>(-22, -14.5, 13.0);
        static const auto r = 1.0;
        if (gx.size() == x.size())
        {
            gx = P * x + q;
        }
        return 0.5 * x.dot(P * x) + x.dot(q) + r;
    };
    auto function = make_function(3, convexity::yes, smoothness::yes, 0.0, lambda);
    UTEST_CHECK(function.constrain(minimum_t{-1.0, 0}));
    UTEST_CHECK(function.constrain(minimum_t{-1.0, 1}));
    UTEST_CHECK(function.constrain(minimum_t{-1.0, 2}));
    UTEST_CHECK(function.constrain(maximum_t{+1.0, 0}));
    UTEST_CHECK(function.constrain(maximum_t{+1.0, 1}));
    UTEST_CHECK(function.constrain(maximum_t{+1.0, 2}));

    check_gradient(function);
    check_convexity(function);
    check_penalties(function, convexity::yes, smoothness::yes, 0.0);

    const auto fbest = -21.625;
    const auto xbest = make_vector<scalar_t>(1.0, 0.5, -1.0);
    check_penalty_solver(function, xbest, fbest, 1e-1);
}

UTEST_END_MODULE()
