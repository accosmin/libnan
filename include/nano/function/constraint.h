#pragma once

#include <memory>
#include <nano/arch.h>
#include <nano/tensor.h>
#include <variant>

namespace nano
{
class NANO_PUBLIC function_t;
using rfunction_t  = std::unique_ptr<function_t>;
using rfunctions_t = std::vector<rfunction_t>;

namespace constraint
{
struct NANO_PUBLIC euclidean_ball_t
{
    vector_t m_origin;
    scalar_t m_radius{0.0};
};

struct NANO_PUBLIC linear_t
{
    vector_t m_q;
    scalar_t m_r{0.0};
};

struct NANO_PUBLIC quadratic_t
{
    matrix_t m_P;
    vector_t m_q;
    scalar_t m_r{0.0};
};

struct NANO_PUBLIC functional_t
{
    functional_t() = default;
    explicit functional_t(const function_t&);
    explicit functional_t(rfunction_t&& function);

    ~functional_t()                       = default;
    functional_t(functional_t&&) noexcept = default;
    functional_t(const functional_t&);
    functional_t& operator=(functional_t&&) noexcept = default;
    functional_t& operator=(const functional_t&);

    rfunction_t m_function;
};

///
/// \brief equality constraint: h(x) = ||x - origin||^2 - radius^2 = 0
///
struct NANO_PUBLIC euclidean_ball_equality_t : euclidean_ball_t
{
};

///
/// \brief inequality constraint: g(x) = ||x - origin||^2 - radius^2 <= 0
///
struct NANO_PUBLIC euclidean_ball_inequality_t : euclidean_ball_t
{
};

///
/// \brief equality constraint: h(x) = x(dimension) - constant = 0
///
struct NANO_PUBLIC constant_t
{
    scalar_t      m_value{0.0};
    tensor_size_t m_dimension{-1};
};

///
/// \brief inequality constraint: g(x) = constant - x(dimension) <= 0
///
struct NANO_PUBLIC minimum_t : constant_t
{
};

///
/// \brief inequality constraint: g(x) = x(dimension) - constant <= 0
///
struct NANO_PUBLIC maximum_t : constant_t
{
};

///
/// \brief equality constraint: h(x) = q.dot(x) + r = 0
///
struct NANO_PUBLIC linear_equality_t : linear_t
{
};

///
/// \brief inequality constraint: g(x) = q.dot(x) + r <= 0
///
struct NANO_PUBLIC linear_inequality_t : linear_t
{
};

///
/// \brief equality constraint: h(x) = 1/2 * x.dot(P * x) + q.dot(x) + r = 0
///
struct NANO_PUBLIC quadratic_equality_t : quadratic_t
{
};

///
/// \brief inequality constraint: q(x) = 1/2 * x.dot(P * x) + q.dot(x) + r = 0
///
struct NANO_PUBLIC quadratic_inequality_t : quadratic_t
{
};

///
/// \brief equality constraint: h(x) = 0
///
struct NANO_PUBLIC functional_equality_t : functional_t
{
    using functional_t::functional_t; // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
};

///
/// \brief inequality constraint: g(x) <= 0
///
struct NANO_PUBLIC functional_inequality_t : functional_t
{
    using functional_t::functional_t; // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
};
} // namespace constraint

///
/// \brief models a constraint that can be applied to an objective function.
///
/// NB: the default constraint is by construction invalid.
///
using constraint_t = std::variant<constraint::constant_t,                  ///<
                                  constraint::minimum_t,                   ///<
                                  constraint::maximum_t,                   ///<
                                  constraint::euclidean_ball_equality_t,   ///<
                                  constraint::euclidean_ball_inequality_t, ///<
                                  constraint::linear_equality_t,           ///<
                                  constraint::linear_inequality_t,         ///<
                                  constraint::quadratic_equality_t,        ///<
                                  constraint::quadratic_inequality_t,      ///<
                                  constraint::functional_equality_t,       ///<
                                  constraint::functional_inequality_t>;    ///<

using constraints_t = std::vector<constraint_t>;

///
/// \brief returns whether the given constraint is convex.
///
NANO_PUBLIC bool convex(const constraint_t&);

///
/// \brief returns whether the given constraint is smooth.
///
NANO_PUBLIC bool smooth(const constraint_t&);

///
/// \brief returns whether the strong convexity coefficient of the given constraint.
///
NANO_PUBLIC scalar_t strong_convexity(const constraint_t&);

///
/// \brief returns how much a point violates the given constraint (the larger, the worse).
///
NANO_PUBLIC scalar_t valid(const constraint_t&, vector_cmap_t x);

///
/// \brief evaluate the given constraint's function value at the given point
///     (and its gradient or sub-gradient if not smooth).
///
NANO_PUBLIC scalar_t vgrad(const constraint_t&, vector_cmap_t x, vector_map_t gx = vector_map_t{});

///
/// \brief returns true if the given function and constraint are compatible.
///
NANO_PUBLIC bool compatible(const constraint_t&, const function_t&);

///
/// \brief returns true if the given constraint is an equality constraint.
///
NANO_PUBLIC bool is_equality(const constraint_t&);

///
/// \brief returns true if the given constraint is a linear (equality or inequality) constraint.
///
NANO_PUBLIC bool is_linear(const constraint_t&);

///
/// \brief returns the number of equality constraints.
///
NANO_PUBLIC tensor_size_t n_equalities(const function_t&);
NANO_PUBLIC tensor_size_t n_equalities(const constraints_t&);

///
/// \brief returns the number of inequality constraints.
///
NANO_PUBLIC tensor_size_t n_inequalities(const function_t&);
NANO_PUBLIC tensor_size_t n_inequalities(const constraints_t&);
} // namespace nano
