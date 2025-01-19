#pragma once

#include <solver/gsample/perturbation.h>

namespace nano::gsample
{
class lsearch_t
{
public:
    lsearch_t(const tensor_size_t n, const configurable_t& configurable, const string_t& basename)
        : m_beta(configurable.parameter(basename + "lsearch_beta").value<scalar_t>())
        , m_gamma(configurable.parameter(basename + "lsearch_gamma").value<scalar_t>())
        , m_max_iters(configurable.parameter(basename + "lsearch_max_iters").value<tensor_size_t>())
        , m_perturbation(n, configurable.parameter(basename + "lsearch_perturb_c").value<scalar_t>())
    {
    }

    template <class thessian>
    scalar_t step(vector_t& x, const vector_t& g, solver_state_t& state, const thessian& H)
    {
        const auto& function = state.function();

        const auto d  = g + m_perturbation.generate(state, g);
        const auto df = m_beta * g.dot(H * g);

        auto t = 1.0;
        if (auto fx = function(x = state.x() - t * d); fx < state.fx() - t * df)
        {
            // doubling phase
            for (auto iters = 0; iters < m_max_iters; ++iters)
            {
                if (t /= m_gamma, fx = function(x = state.x() - t * d); fx >= state.fx() - t * df)
                {
                    t *= m_gamma;
                    state.update(x = state.x() - t * d);
                    return t;
                }
            }
        }
        else
        {
            // bisection phase
            for (auto iters = 0; iters < m_max_iters; ++iters)
            {
                if (t *= m_gamma, fx = function(x = state.x() - t * d); fx < state.fx() - t * df)
                {
                    state.update(x = state.x() - t * d);
                    return t;
                }
            }
        }

        return 0.0;
    }

private:
    // attributes
    scalar_t       m_beta{1e-8};
    scalar_t       m_gamma{0.5};
    tensor_size_t  m_max_iters{50};
    perturbation_t m_perturbation;
};
} // namespace nano::gsample
