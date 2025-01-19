#include <nano/core/strutil.h>
#include <nano/function/penalty.h>

using namespace nano;

namespace
{
auto is_linear_equality(const constraint_t& constraint)
{
    return std::get_if<constraint::constant_t>(&constraint) != nullptr ||
           std::get_if<constraint::linear_equality_t>(&constraint) != nullptr;
}

auto convex(const function_t& function)
{
    const auto op = [](const auto& ct) { return ::nano::convex(ct) && (!is_equality(ct) || is_linear_equality(ct)); };
    return (function.convex() && std::all_of(function.constraints().begin(), function.constraints().end(), op))
             ? convexity::yes
             : convexity::no;
}

auto smooth(const function_t& function)
{
    const auto op = [](const auto& constraint) { return ::nano::smooth(constraint); };
    return (function.smooth() && std::all_of(function.constraints().begin(), function.constraints().end(), op))
             ? smoothness::yes
             : smoothness::no;
}

template <class toperator>
auto penalty_vgrad(const function_t& function, vector_cmap_t x, vector_map_t gx, const toperator& op)
{
    auto fx = function(x, gx);

    for (const auto& constraint : function.constraints())
    {
        auto       gc = vector_t{gx.size()}; // FIXME: is this allocation really necessary?!
        const auto fc = ::nano::vgrad(constraint, x, gc);
        const auto eq = is_equality(constraint);

        if (eq || fc > 0.0)
        {
            fx += op(fc, gc);
        }
    }

    return fx;
}
} // namespace

penalty_function_t::penalty_function_t(const function_t& function, const char* const prefix)
    : function_t(scat(prefix, function.name()), function.size())
    , m_function(function)
{
    // NB: cannot estimate the strong convexity coefficient in general!
    strong_convexity(function.strong_convexity());

    // NB: no constraints are needed for the penalty function!
}

penalty_function_t& penalty_function_t::penalty(scalar_t penalty)
{
    m_penalty = penalty;
    return *this;
}

linear_penalty_function_t::linear_penalty_function_t(const function_t& function)
    : penalty_function_t(function, "linear-penalty::")
{
    convex(::convex(function));
    smooth(function.constraints().empty() ? ::smooth(function) : smoothness::no);
}

rfunction_t linear_penalty_function_t::clone() const
{
    return std::make_unique<linear_penalty_function_t>(*this);
}

scalar_t linear_penalty_function_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    assert(x.size() == size());

    const auto op = [&](const scalar_t fc, const vector_t& gc)
    {
        if (gx.size() == x.size())
        {
            gx += penalty() * (fc >= 0.0 ? +1.0 : -1.0) * gc;
        }
        return penalty() * std::fabs(fc);
    };

    return penalty_vgrad(function(), x, gx, op);
}

quadratic_penalty_function_t::quadratic_penalty_function_t(const function_t& function)
    : penalty_function_t(function, "quadratic-penalty::")
{
    convex(::convex(function));
    smooth(::smooth(function));
}

rfunction_t quadratic_penalty_function_t::clone() const
{
    return std::make_unique<quadratic_penalty_function_t>(*this);
}

scalar_t quadratic_penalty_function_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    assert(x.size() == size());

    const auto op = [&](const scalar_t fc, const vector_t& gc)
    {
        if (gx.size() == x.size())
        {
            gx += penalty() * 2.0 * fc * gc;
        }
        return penalty() * fc * fc;
    };

    return penalty_vgrad(function(), x, gx, op);
}

augmented_lagrangian_function_t::augmented_lagrangian_function_t(const function_t& function, const vector_t& lambda,
                                                                 const vector_t& miu)
    : penalty_function_t(function, "augmented-lagrangian::")
    , m_lambda(lambda)
    , m_miu(miu)
{
    assert(m_lambda.size() == function.n_equalities());
    assert(m_miu.size() == function.n_inequalities());

    convex(::convex(function));
    smooth(::smooth(function));
}

rfunction_t augmented_lagrangian_function_t::clone() const
{
    return std::make_unique<augmented_lagrangian_function_t>(*this);
}

scalar_t augmented_lagrangian_function_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    assert(x.size() == size());

    auto fx      = function()(x, gx);
    auto ilambda = tensor_size_t{0};
    auto imiu    = tensor_size_t{0};
    auto gc      = vector_t{gx.size()};

    for (const auto& constraint : function().constraints())
    {
        const auto ro = penalty();
        const auto fc = ::nano::vgrad(constraint, x, gc);
        const auto eq = is_equality(constraint);
        const auto mu = eq ? m_lambda(ilambda++) : m_miu(imiu++);

        if (eq || (fc + mu / ro > 0.0))
        {
            fx += 0.5 * ro * (fc + mu / ro) * (fc + mu / ro);
            if (gx.size() == x.size())
            {
                gx += ro * (fc + mu / ro) * gc;
            }
        }
    }

    return fx;
}
