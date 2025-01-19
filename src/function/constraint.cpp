#include <nano/core/overloaded.h>
#include <nano/function/constraint.h>
#include <nano/function/util.h>

using namespace nano;
using namespace nano::constraint;

namespace
{
auto smooth(const euclidean_ball_t&)
{
    return true;
}

auto smooth(const linear_t&)
{
    return true;
}

auto smooth(const constant_t&)
{
    return true;
}

auto smooth(const quadratic_t&)
{
    return true;
}

auto smooth(const functional_t& constraint)
{
    return constraint.m_function->smooth();
}

auto strong_convexity(const euclidean_ball_t&)
{
    return 2.0;
}

auto strong_convexity(const linear_t&)
{
    return 0.0;
}

auto strong_convexity(const constant_t&)
{
    return 0.0;
}

auto strong_convexity(const quadratic_t& constraint)
{
    return nano::strong_convexity(constraint.m_P);
}

scalar_t strong_convexity(const functional_t& constraint)
{
    return constraint.m_function->strong_convexity();
}

auto vgrad(const euclidean_ball_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    if (gx.size() == x.size())
    {
        gx = 2.0 * (x - constraint.m_origin);
    }
    return (x - constraint.m_origin).squaredNorm() - constraint.m_radius * constraint.m_radius;
}

auto vgrad(const linear_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    if (gx.size() == x.size())
    {
        gx = constraint.m_q;
    }
    return constraint.m_q.dot(x) + constraint.m_r;
}

auto vgrad(const quadratic_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    const auto P = constraint.m_P.matrix();
    const auto q = constraint.m_q.vector();
    if (gx.size() == x.size())
    {
        gx = P * x.vector() + q;
    }
    return 0.5 * x.vector().dot(P * x.vector()) + q.dot(x.vector()) + constraint.m_r;
}

auto vgrad(const minimum_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    if (gx.size() == x.size())
    {
        gx.full(0.0)(constraint.m_dimension) = -1.0;
    }
    return constraint.m_value - x(constraint.m_dimension);
}

auto vgrad(const maximum_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    if (gx.size() == x.size())
    {
        gx.full(0.0)(constraint.m_dimension) = +1.0;
    }
    return x(constraint.m_dimension) - constraint.m_value;
}

auto vgrad(const constant_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    if (gx.size() == x.size())
    {
        gx.full(0.0)(constraint.m_dimension) = +1.0;
    }
    return x(constraint.m_dimension) - constraint.m_value;
}

auto vgrad(const functional_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    return (*constraint.m_function)(x, gx);
}

bool convex(const euclidean_ball_t&)
{
    return true;
}

bool convex(const linear_t&)
{
    return true;
}

bool convex(const constant_t&)
{
    return true;
}

bool convex(const quadratic_t& constraint)
{
    return nano::is_convex(constraint.m_P);
}

bool convex(const functional_t& constraint)
{
    return constraint.m_function->convex();
}

auto valid(const constant_t& constraint, vector_cmap_t x)
{
    return std::fabs(constraint.m_value - x(constraint.m_dimension));
}

auto valid(const minimum_t& constraint, vector_cmap_t x)
{
    return std::max(constraint.m_value - x(constraint.m_dimension), 0.0);
}

auto valid(const maximum_t& constraint, vector_cmap_t x)
{
    return std::max(x(constraint.m_dimension) - constraint.m_value, 0.0);
}

auto valid(const euclidean_ball_equality_t& constraint, vector_cmap_t x)
{
    return std::fabs(::vgrad(constraint, x));
}

auto valid(const euclidean_ball_inequality_t& constraint, vector_cmap_t x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

auto valid(const linear_equality_t& constraint, vector_cmap_t x)
{
    return std::fabs(::vgrad(constraint, x));
}

auto valid(const linear_inequality_t& constraint, vector_cmap_t x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

auto valid(const quadratic_equality_t& constraint, vector_cmap_t x)
{
    return std::fabs(::vgrad(constraint, x));
}

auto valid(const quadratic_inequality_t& constraint, vector_cmap_t x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

auto valid(const functional_equality_t& constraint, vector_cmap_t x)
{
    return std::fabs(::vgrad(constraint, x));
}

auto valid(const functional_inequality_t& constraint, vector_cmap_t x)
{
    return std::max(::vgrad(constraint, x), 0.0);
}

auto compatible(const function_t& function, const euclidean_ball_t& constraint)
{
    return constraint.m_origin.size() == function.size() && constraint.m_radius > 0.0;
}

auto compatible(const function_t& function, const linear_t& constraint)
{
    return constraint.m_q.size() == function.size();
}

auto compatible(const function_t& function, const constant_t& constraint)
{
    return constraint.m_dimension >= 0 && constraint.m_dimension < function.size();
}

auto compatible(const function_t& function, const quadratic_t& constraint)
{
    return constraint.m_P.rows() == function.size() && constraint.m_P.cols() == function.size() &&
           constraint.m_q.size() == function.size();
}

auto compatible(const function_t& function, const functional_t& constraint)
{
    return static_cast<bool>(constraint.m_function) && constraint.m_function->size() == function.size();
}
} // namespace

functional_t::functional_t(const function_t& function)
    : m_function(function.clone())
{
}

functional_t::functional_t(rfunction_t&& function)
    : m_function(std::move(function))
{
}

functional_t::functional_t(const functional_t& other)
    : m_function(other.m_function->clone())
{
}

functional_t& functional_t::operator=(const functional_t& other)
{
    if (this != &other)
    {
        m_function = other.m_function->clone();
    }
    return *this;
}

scalar_t nano::strong_convexity(const constraint_t& constraint)
{
    return std::visit(overloaded{[&](const euclidean_ball_t& ct) { return ::strong_convexity(ct); },
                                 [&](const linear_t& ct) { return ::strong_convexity(ct); },
                                 [&](const constant_t& ct) { return ::strong_convexity(ct); },
                                 [&](const quadratic_t& ct) { return ::strong_convexity(ct); },
                                 [&](const functional_t& ct) { return ::strong_convexity(ct); }},
                      constraint);
}

scalar_t nano::vgrad(const constraint_t& constraint, vector_cmap_t x, vector_map_t gx)
{
    return std::visit(
        // clang-format off
        overloaded{[&](const euclidean_ball_t& ct) { return ::vgrad(ct, x, gx); },
                   [&](const linear_t& ct) { return ::vgrad(ct, x, gx); },
                   [&](const constant_t& ct) { return ::vgrad(ct, x, gx); },
                   [&](const maximum_t& ct) { return ::vgrad(ct, x, gx); },
                   [&](const minimum_t& ct) { return ::vgrad(ct, x, gx); },
                   [&](const quadratic_t& ct) { return ::vgrad(ct, x, gx); },
                   [&](const functional_t& ct) { return ::vgrad(ct, x, gx); }},
        // clang-format on
        constraint);
}

scalar_t nano::valid(const constraint_t& constraint, vector_cmap_t x)
{
    return std::visit(
        // clang-format off
        overloaded{[&](const minimum_t& ct) { return ::valid(ct, x); },
                   [&](const maximum_t& ct) { return ::valid(ct, x); },
                   [&](const constant_t& ct) { return ::valid(ct, x); },
                   [&](const euclidean_ball_equality_t& ct) { return ::valid(ct, x); },
                   [&](const euclidean_ball_inequality_t& ct) { return ::valid(ct, x); },
                   [&](const linear_equality_t& ct) { return ::valid(ct, x); },
                   [&](const linear_inequality_t& ct) { return ::valid(ct, x); },
                   [&](const quadratic_equality_t& ct) { return ::valid(ct, x); },
                   [&](const quadratic_inequality_t& ct) { return ::valid(ct, x); },
                   [&](const functional_equality_t& ct) { return ::valid(ct, x); },
                   [&](const functional_inequality_t& ct) { return ::valid(ct, x); }},
        // clang-format on
        constraint);
}

bool nano::convex(const constraint_t& constraint)
{
    return std::visit(
        // clang-format off
        overloaded{[&](const euclidean_ball_t& ct) { return ::convex(ct); },
                   [&](const linear_t& ct) { return ::convex(ct); },
                   [&](const constant_t& ct) { return ::convex(ct); },
                   [&](const quadratic_t& ct) { return ::convex(ct); },
                   [&](const functional_t& ct) { return ::convex(ct); }},
        // clang-format on
        constraint);
}

bool nano::smooth(const constraint_t& constraint)
{
    return std::visit(
        // clang-format off
        overloaded{[&](const euclidean_ball_t& ct) { return ::smooth(ct); },
                   [&](const linear_t& ct) { return ::smooth(ct); },
                   [&](const constant_t& ct) { return ::smooth(ct); },
                   [&](const quadratic_t& ct) { return ::smooth(ct); },
                   [&](const functional_t& ct) { return ::smooth(ct); }},
        // clang-format on
        constraint);
}

bool nano::compatible(const constraint_t& constraint, const function_t& function)
{
    return std::visit(overloaded{[&](const euclidean_ball_t& ct) { return ::compatible(function, ct); },
                                 [&](const linear_t& ct) { return ::compatible(function, ct); },
                                 [&](const constant_t& ct) { return ::compatible(function, ct); },
                                 [&](const quadratic_t& ct) { return ::compatible(function, ct); },
                                 [&](const functional_t& ct) { return ::compatible(function, ct); }},
                      constraint);
}

bool nano::is_equality(const constraint_t& constraint)
{
    return std::visit(overloaded{[](const constant_t&) { return true; },                   ///<
                                 [](const minimum_t&) { return false; },                   ///<
                                 [](const maximum_t&) { return false; },                   ///<
                                 [](const linear_equality_t&) { return true; },            ///<
                                 [](const linear_inequality_t&) { return false; },         ///<
                                 [](const euclidean_ball_equality_t&) { return true; },    ///<
                                 [](const euclidean_ball_inequality_t&) { return false; }, ///<
                                 [](const quadratic_equality_t&) { return true; },         ///<
                                 [](const quadratic_inequality_t&) { return false; },      ///<
                                 [](const functional_equality_t&) { return true; },        ///<
                                 [](const functional_inequality_t&) { return false; }},    ///<
                      constraint);
}

bool nano::is_linear(const constraint_t& constraint)
{
    return std::visit(overloaded{[](const constant_t&) { return true; },          ///<
                                 [](const minimum_t&) { return true; },           ///<
                                 [](const maximum_t&) { return true; },           ///<
                                 [](const linear_equality_t&) { return true; },   ///<
                                 [](const linear_inequality_t&) { return true; }, ///<
                                 [](const euclidean_ball_t&) { return false; },   ///<
                                 [](const quadratic_t&) { return false; },        ///<
                                 [](const functional_t&) { return false; }},      ///<
                      constraint);
}

tensor_size_t nano::n_equalities(const function_t& function)
{
    return n_equalities(function.constraints());
}

tensor_size_t nano::n_equalities(const constraints_t& constraints)
{
    const auto op = [](const auto& constraint) { return is_equality(constraint); };
    return std::count_if(std::begin(constraints), std::end(constraints), op);
}

tensor_size_t nano::n_inequalities(const function_t& function)
{
    return n_inequalities(function.constraints());
}

tensor_size_t nano::n_inequalities(const constraints_t& constraints)
{
    const auto op = [](const auto& constraint) { return !is_equality(constraint); };
    return std::count_if(std::begin(constraints), std::end(constraints), op);
}
