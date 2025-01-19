#include <nano/solver/track.h>

using namespace nano;

solver_track_t::solver_track_t(vector_t x, const scalar_t fx)
    : m_prev_x(std::move(x))
    , m_prev_fx(fx)
{
}

void solver_track_t::update(vector_t x, const scalar_t fx)
{
    const auto df = m_prev_fx - fx;
    const auto dx = (m_prev_x - x).lpNorm<Eigen::Infinity>() / std::max(1.0, m_prev_x.lpNorm<Eigen::Infinity>());

    m_history_df.push_back(df);
    m_history_dx.push_back(dx);

    m_prev_x  = std::move(x);
    m_prev_fx = fx;
}

scalar_t solver_track_t::value_test_unconstrained(const tensor_size_t patience) const
{
    assert(m_history_df.size() == m_history_dx.size());

    auto ii = m_history_df.size();
    auto dd = std::numeric_limits<scalar_t>::max();
    for (size_t it = m_history_df.size(); it > 0U; --it)
    {
        const auto df = m_history_df[it - 1U];
        const auto dx = m_history_dx[it - 1U];

        if (df > 0.0)
        {
            dd = std::max(dx, df);
            ii = it - 1U;
            break;
        }
    }

    // no improvement ever recorded, stop if enough iterations have passed
    if (ii == m_history_df.size())
    {
        return m_history_df.size() >= static_cast<size_t>(patience) ? 0.0 : dd;
    }

    // the last improvement was in the recent iterations
    else if (ii + static_cast<size_t>(patience) >= m_history_df.size())
    {
        return dd;
    }

    // the last improvement was not in the recent iterations, stop with potential convergence status
    else
    {
        return 0.0;
    }
}

scalar_t solver_track_t::value_test_constrained(const tensor_size_t patience) const
{
    assert(m_history_df.size() == m_history_dx.size());

    const auto size  = static_cast<tensor_size_t>(m_history_dx.size());
    const auto begin = size >= patience ? (size - patience) : tensor_size_t{0};

    const auto itmax = std::max_element(m_history_dx.begin() + begin, m_history_dx.end());
    if (itmax != m_history_dx.end())
    {
        return *itmax;
    }
    else
    {
        return std::numeric_limits<scalar_t>::max();
    }
}
