#include <nano/function.h>
#include <nano/core/numeric.h>
#include <nano/core/strutil.h>

using namespace nano;

function_t::function_t(string_t name, tensor_size_t size) :
    m_name(std::move(name)),
    m_size(size)
{
}

void function_t::convex(bool convex)
{
    m_convex = convex;
}

void function_t::smooth(bool smooth)
{
    m_smooth = smooth;
}

void function_t::summands(tensor_size_t summands)
{
    m_summands = summands;
}

void function_t::strong_convexity(scalar_t sconvexity)
{
    m_sconvexity = sconvexity;
}

scalar_t function_t::grad_accuracy(const vector_t& x) const
{
    assert(x.size() == size());

    const auto n = size();

    vector_t gx(n);
    vector_t gx_approx(n);
    vector_t xp = x, xn = x;

    // analytical gradient
    const auto fx = vgrad(x, &gx);
    assert(gx.size() == size());

    // finite-difference approximated gradient
    //      see "Numerical optimization", Nocedal & Wright, 2nd edition, p.197
    const auto dx = epsilon2<scalar_t>();
    for (auto i = 0; i < n; i ++)
    {
        if (i > 0)
        {
            xp(i - 1) = x(i - 1);
            xn(i - 1) = x(i - 1);
        }
        xp(i) = x(i) + dx * std::max(scalar_t{1}, std::fabs(x(i)));
        xn(i) = x(i) - dx * std::max(scalar_t{1}, std::fabs(x(i)));

        const auto dfi = vgrad(xp, nullptr) - vgrad(xn, nullptr);
        const auto dxi = xp(i) - xn(i);
        gx_approx(i) = dfi / dxi;

        assert(std::isfinite(gx(i)));
        assert(std::isfinite(gx_approx(i)));
    }

    return (gx - gx_approx).lpNorm<Eigen::Infinity>() / (1 + std::fabs(fx));
}

bool function_t::is_convex(const vector_t& x1, const vector_t& x2, const int steps) const
{
    assert(steps > 2);
    assert(x1.size() == size());
    assert(x2.size() == size());

    const auto f1 = vgrad(x1, nullptr);
    assert(std::isfinite(f1));

    const auto f2 = vgrad(x2, nullptr);
    assert(std::isfinite(f2));

    const auto eps = epsilon1<scalar_t>();
    const auto dx = (x1 - x2).dot(x1 - x2);

    for (int step = 1; step < steps; step ++)
    {
        const auto t1 = scalar_t(step) / scalar_t(steps);
        const auto t2 = 1.0 - t1;

        if (vgrad(t1 * x1 + t2 * x2) > t1 * f1 + t2 * f2 - t1 * t2 * m_sconvexity * 0.5 * dx + eps)
        {
            return false;
        }
    }

    return true;
}

string_t function_t::name() const
{
    return scat(m_name, "[", size(), "D]");
}
