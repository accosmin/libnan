#pragma once

#include <nano/solver/state.h>

namespace nano
{
///
/// \brief Nesterov accelerated schemes useful in algorithms to minimize:
///     * initially designed for strong convex smooth functions - see (1)
///     * and later for non-smooth convex functions - see (2, 3, 4).
///
/// see (1) "A method for solving a convex programming problem with convergence rage O(1/k^2)", by Nesterov, 1983
/// see (2) "New proximal point algorithm for convex minimization", by Guler, 1992
/// see (3) "Proximal bundle algorithms for nonsmooth convex optimization via fast gradient smooth methods",
///         by Ouorou, 2020
/// see (4) "Fast proximal algorithms for nonsmooth convex optimization", by Ouorou, 2020
///
template <class tderived>
class nesterov_sequence_t
{
public:
    explicit nesterov_sequence_t(const solver_state_t& state)
        : m_x(state.x())
        , m_y(state.x())
    {
    }

    void reset() { m_lambda = 1.0; }

    scalar_t lambda() const { return m_lambda; }

    scalar_t update()
    {
        m_lambda = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * m_lambda * m_lambda));
        return m_lambda;
    }

    const vector_t& update(const vector_cmap_t z)
    {
        const auto [ak, bk] = static_cast<tderived*>(this)->make_alpha_beta();
        m_x                 = z + ak * (z - m_y) + bk * (z - m_x);
        m_y                 = z;
        return m_x;
    }

private:
    // attributes
    scalar_t m_lambda{1.0}; ///<
    vector_t m_x;           ///<
    vector_t m_y;           ///<
};

///
/// \brief generate Nesterov-like sequence - see (1, 2, 3, 4):
///     x_k+1 = y_k+1 + alpha_k * (y_k+1 - y_k).
///
class nesterov_sequence1_t final : public nesterov_sequence_t<nesterov_sequence1_t>
{
public:
    explicit nesterov_sequence1_t(const solver_state_t&);

    std::tuple<scalar_t, scalar_t> make_alpha_beta();

    static const char* str() { return "1"; }
};

///
/// \brief generate Nesterov-like sequence - see (2, 3, 4):
///     x_k+1 = y_k+1 + alpha_k * (y_k+1 - y_k) + beta_k * (y_k+1 - x_k).
///
class nesterov_sequence2_t final : public nesterov_sequence_t<nesterov_sequence2_t>
{
public:
    explicit nesterov_sequence2_t(const solver_state_t&);

    std::tuple<scalar_t, scalar_t> make_alpha_beta();

    static const char* str() { return "2"; }
};
} // namespace nano
