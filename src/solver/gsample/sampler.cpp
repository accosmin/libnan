#include <nano/core/sampling.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>
#include <nano/tensor/algorithm.h>
#include <solver/gsample/sampler.h>

using namespace nano;
using namespace nano::gsample;

sampler_t::sampler_t(const tensor_size_t n)
    : m_X(2 * n + 1, n)
    , m_G(2 * n + 1, n)
    , m_solver(solver_t::all().get("ipm"))
{
}

quadratic_program_t sampler_t::make_program(const tensor_size_t p)
{
    auto program = quadratic_program_t{"gsample-qp", matrix_t{matrix_t::zero(p, p)}, vector_t{vector_t::zero(p)}};

    critical(program.variable() >= 0.0);
    critical(vector_t::constant(p, 1.0) * program.variable() == 1.0);

    return program;
}

fixed_sampler_t::fixed_sampler_t(const tensor_size_t n)
    : sampler_t(n)
    , m_program(make_program(m_G.rows()))
{
}

void fixed_sampler_t::sample(const solver_state_t& state, const scalar_t epsilon)
{
    const auto m = m_G.rows() - 1;

    m_psize  = 0;
    auto rng = make_rng();
    for (tensor_size_t i = 0; i < m; ++i, ++m_psize)
    {
        sample_from_ball(state.x(), epsilon, m_X.tensor(i), rng);
        assert((state.x() - m_X.tensor(i)).lpNorm<2>() <= epsilon + std::numeric_limits<scalar_t>::epsilon());
        state.function()(m_X.tensor(i), m_G.tensor(i));
    }

    m_X.tensor(m) = state.x();
    m_G.tensor(m) = state.gx();
    ++m_psize;

    assert(m_psize == m_X.rows());
    assert(m_psize == m_G.rows());
}

adaptive_sampler_t::adaptive_sampler_t(const tensor_size_t n)
    : sampler_t(n)
{
}

void adaptive_sampler_t::sample(const solver_state_t& state, const scalar_t epsilon)
{
    const auto p    = m_X.rows();
    const auto n    = m_X.cols();
    const auto phat = std::max(n / 10, tensor_size_t{1});

    // remove previously selected points outside the current ball
    const auto op1 = [&](const tensor_size_t i) { return (state.x() - m_X.tensor(i)).lpNorm<2>() > epsilon; };
    m_psize        = nano::remove_if(op1, m_X.slice(0, m_psize), m_G.slice(0, m_psize));

    // NB: to make sure at most `p` samples are used!
    const auto op2 = [&](const tensor_size_t i) { return i < (m_psize + 1 + phat - p); };
    m_psize        = nano::remove_if(op2, m_X.slice(0, m_psize), m_G.slice(0, m_psize));
    assert(m_psize + 1 + phat <= p);

    // current point (center of the ball)
    m_X.tensor(m_psize) = state.x();
    m_G.tensor(m_psize) = state.gx();
    ++m_psize;

    // new samples
    auto rng = make_rng();
    for (tensor_size_t i = 0; i < phat; ++i, ++m_psize)
    {
        assert(m_psize < p);
        sample_from_ball(state.x(), epsilon, m_X.tensor(m_psize), rng);
        state.function()(m_X.tensor(m_psize), m_G.tensor(m_psize));
    }

    for (tensor_size_t i = 0; i < m_psize; ++i)
    {
        assert((state.x() - m_X.tensor(i)).lpNorm<2>() <= epsilon + std::numeric_limits<scalar_t>::max());
    }
}
