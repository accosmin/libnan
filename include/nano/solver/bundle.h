#pragma once

#include <nano/configurable.h>
#include <nano/logger.h>
#include <nano/solver/state.h>
#include <nano/tensor/algorithm.h>

namespace nano
{
///
/// \brief models the bundle of sub-gradients as used by penalized (proximal) bundle algorithms.
///
/// see (1) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (2) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
///
/// NB: the bundle is kept small by:
///     - first removing all inactive constraints and
///     - then the ones with the largest approximation error if needed - see (2).
///
class NANO_PUBLIC bundle_t
{
public:
    ///
    /// \brief constructor
    ///
    bundle_t(const solver_state_t&, tensor_size_t max_size);

    ///
    /// \brief setup the default configuration.
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief construct an empty bundle with the given configuration.
    ///
    static bundle_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current size of the bundle.
    ///
    tensor_size_t size() const { return m_size; }

    ///
    /// \brief return the proximity center.
    ///
    const vector_t& x() const { return m_x; }

    ///
    /// \brief return the sub-gradient at the proximity center.
    ///
    const vector_t& gx() const { return m_gx; }

    ///
    /// \brief return the function value at the proximity center.
    ///
    scalar_t fx() const { return m_fx; }

    ///
    /// \brief return the smeared approximation error, see (1).
    ///
    auto smeared_e() const { return e().dot(alpha()); }

    ///
    /// \brief return the smeared sub-gradient, see (1).
    ///
    auto smeared_s() const { return S().transpose() * alpha(); }

    ///
    /// \brief return the approximation error, see (1).
    ///
    scalar_t delta(const scalar_t miu) const
    {
        const auto delta = smeared_e() + 1.0 / (2.0 * miu) * smeared_s().squaredNorm();
        assert(delta + epsilon1<scalar_t>() >= 0.0);
        return delta;
    }

    ///
    /// \brief return the estimated proximal point.
    ///
    auto proximal(const scalar_t miu) const { return m_x - smeared_s() / miu; }

    ///
    /// \brief change the proximity center to the given point and update the bundle.
    ///
    void moveto(vector_cmap_t y, vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief update the bundle with the given point.
    ///
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy);

    ///
    /// \brief return the solution of the penalized proximal bundle problem.
    ///
    void solve(scalar_t miu, const logger_t&);

    ///
    /// \brief return true if converged wrt the smeared approximation error, see (1).
    /// FIXME: citation here, but at least use it consistently across proximal bundle algorithms.
    ///
    bool econverged(scalar_t epsilon) const;

    ///
    /// \brief return true if converged wrt the smeared sub-gradient, see (1).
    /// FIXME: citation here, but at least use it consistently across proximal bundle algorithms.
    ///
    bool sconverged(scalar_t epsilon) const;

private:
    tensor_size_t dims() const { return m_bundleS.size<1>(); }

    tensor_size_t capacity() const { return m_alphas.size(); }

    vector_cmap_t alpha() const { return m_alphas.slice(0, size()); }

    matrix_cmap_t S() const { return m_bundleS.slice(0, size()); }

    vector_cmap_t e() const
    {
        auto e = m_bundleE.slice(0, size());
        assert(e.min() + epsilon1<scalar_t>() > 0.0);
        return e;
    }

    template <class toperator>
    tensor_size_t remove_if(const toperator& op)
    {
        return nano::remove_if(op, m_bundleE.slice(0, m_size), m_bundleS.slice(0, m_size), m_alphas.slice(0, m_size));
    }

    void delete_inactive(scalar_t epsilon);
    void delete_largest(tensor_size_t count);

    void store_aggregate();
    void append_aggregate();
    void append(vector_cmap_t y, vector_cmap_t gy, scalar_t fy, bool serious_step);

    // attributes
    tensor_size_t m_size{0}; ///< bundle: number of points
    matrix_t      m_bundleS; ///< bundle: sub-gradients (size, dims)
    vector_t      m_bundleE; ///< bundle: linearized errors (size)
    vector_t      m_alphas;  ///< optimal Lagrange multipliers (size)
    vector_t      m_x;       ///< proximal center (dims)
    vector_t      m_gx;      ///< function gradient at the proximal center (dims)
    scalar_t      m_fx;      ///< function value at the proximal center
};
} // namespace nano
