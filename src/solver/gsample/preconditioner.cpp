#include <nano/function/util.h>
#include <solver/gsample/preconditioner.h>

using namespace nano;
using namespace nano::gsample;

identity_preconditioner_t::identity_preconditioner_t(const tensor_size_t n)
    : m_W(matrix_t::identity(n, n))
    , m_H(matrix_t::identity(n, n))
{
}

void identity_preconditioner_t::update([[maybe_unused]] const scalar_t)
{
}

void identity_preconditioner_t::update([[maybe_unused]] const sampler_t&, [[maybe_unused]] const solver_state_t&,
                                       [[maybe_unused]] const scalar_t)
{
}

lbfgs_preconditioner_t::lbfgs_preconditioner_t(const tensor_size_t n)
    : m_W(n, n)
    , m_H(n, n)
{
}

void lbfgs_preconditioner_t::update(const scalar_t alpha)
{
    // initialization scalar of the inverse Hessian update
    const auto miu_min = 1e-2; // FIXME: make it a parameter
    const auto miu_max = 1e+3; // FIXME: make it a parameter
    if (alpha < 1.0)
    {
        m_miu = std::min(2.0 * m_miu, miu_max);
    }
    else
    {
        m_miu = std::max(0.5 * m_miu, miu_min);
    }
}

void lbfgs_preconditioner_t::update(const sampler_t& sampler, const solver_state_t& state, const scalar_t epsilon)
{
    const auto n     = m_W.rows();
    const auto gamma = 0.1;   // FIXME: make it a parameter
    const auto sigma = 100.0; // FIXME: make it a parameter

    m_W = (1.0 / m_miu) * matrix_t::identity(n, n);
    m_H = m_miu * matrix_t::identity(n, n);

    for (tensor_size_t i = 0; i < sampler.m_psize; ++i)
    {
        const auto d  = sampler.m_X.tensor(i) - state.x();
        const auto y  = sampler.m_G.tensor(i) - state.gx();
        const auto dy = d.dot(y);

        assert(d.dot(d) <= epsilon + std::numeric_limits<scalar_t>::epsilon());

        if (dy >= gamma * epsilon && y.dot(y) <= sigma * epsilon)
        {
            m_Q = matrix_t::identity(n, n) - (y * d.transpose()) / dy;
            m_W = m_Q.transpose() * m_W;
            m_W = m_W * m_Q;
            m_W += (d * d.transpose()) / dy;

            m_H -= (m_H * d * d.transpose() * m_H) / (d.transpose() * m_H * d);
            m_H += (y * y.transpose()) / dy;
        }
    }

    assert(is_convex(m_W));
    assert(is_convex(m_H));

    assert((m_W * m_H - matrix_t::identity(n, n)).lpNorm<Eigen::Infinity>() < 1e-9);
    assert((m_H * m_W - matrix_t::identity(n, n)).lpNorm<Eigen::Infinity>() < 1e-9);
}
