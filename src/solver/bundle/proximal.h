#pragma once

#include <nano/configurable.h>
#include <nano/solver/state.h>

namespace nano
{
///
/// \brief models the proximal parameter as used by penalized (proximal) bundle algorithms.
///
/// see (1) "A doubly stabilized bundle method for nonsmooth convex optimization", by Oliveira, Solodov, 2013
/// see (2) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (3) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
/// see (4) "Dynamical adjustment of the prox-parameter in bundle methods", by Rey, Sagastizabal, 2002
/// see (5) "Fast proximal algorithms for nonsmooth convex optimization", by Ouorou, 2020
/// see (6) "A NU-algorithm for convex minimization", by Mifflin, Sagastizabal, 2005
///
/// NB: the proximal parameter is initialized and safeguarded following (6) (ch. 6).
///
/// TODO: different proximal update rules:
///     - see (3) - only for descent steps, but with safeguards like in 6
///     - see (1) - for both descent and null steps, but with safeguardes like in 6
///
/// TODO: experiment with variant of not using ghat
///
class NANO_PUBLIC proximal_t
{
public:
    ///
    /// \brief constructor
    ///
    proximal_t(const solver_state_t& state, scalar_t miu0_min, scalar_t miu0_max, scalar_t min_dot_nuv);

    ///
    /// \brief
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief
    ///
    static proximal_t make(const solver_state_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return the current proximal parameter value.
    ///
    scalar_t miu() const;

    ///
    /// \brief update the proximal parameter given a new proximal center.
    ///
    void update(scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn, const vector_t& gn1);
    void update(scalar_t t, const vector_t& xn, const vector_t& xn1, const vector_t& gn, const vector_t& gn1,
                const vector_t& Gn, const vector_t& Gn1);

private:
    void safeguard(scalar_t miu);

    // attributes
    scalar_t m_miu0{1.0};        ///< initial proximal parameter value
    scalar_t m_miu{1.0};         ///< current proximal parameter value
    scalar_t m_min_dot_nuv{0.0}; ///< minimum nu.dot(v) to accept to adjust the proximal parameter, see (2) or (3)
};
} // namespace nano
