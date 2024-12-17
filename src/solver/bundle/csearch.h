#pragma once

#include <solver/bundle/bundle.h>

namespace nano
{
enum class csearch_status : uint8_t
{
    failed,
    max_iters,
    converged,
    null_step,
    descent_step,
    cutting_plane_step,
};

///
/// \brief curve-search strategy as used by penalized (proximal) bundle algorithms.
///
/// see (1) "Numerical optimization - theoretical and practical aspects", 2nd edition, 2006
/// see (2) "Variable metric bundle methods: from conceptual to implementable forms", by Lemarechal, Sagastizabal, 1997
/// see (3) "Dynamical adjustment of the prox-parameter in bundle methods", by Rey, Sagastizabal, 2002
/// see (4) "A NU-algorithm for convex minimization", by Mifflin, Sagastizabal, 2005
///
/// NB: the implementation follows the notation from (2).
/// NB: the stopping criterion is not clearly given in the references, but some papers specify the following:
///     smeared_error < epsilon * sqrt(N) && smeared_grad < epsilon * sqrt(N).
///
class NANO_PUBLIC csearch_t
{
public:
    struct point_t
    {
        explicit point_t(tensor_size_t dims = 0);

        scalar_t       m_t{1.0};                         ///<
        csearch_status m_status{csearch_status::failed}; ///<
        vector_t       m_y;                              ///<
        vector_t       m_gy;                             ///<
        scalar_t       m_fy{0.0};                        ///<
        vector_t       m_gyhat;                          ///<
        scalar_t       m_fyhat{0.0};                     ///<
    };

    ///
    /// \brief constructor
    ///
    csearch_t(const function_t&, scalar_t m1, scalar_t m2, scalar_t m3, scalar_t m4, scalar_t interpol,
              scalar_t extrapol);

    ///
    /// \brief
    ///
    static void config(configurable_t&, const string_t& prefix);

    ///
    /// \brief
    ///
    static csearch_t make(const function_t&, const configurable_t&, const string_t& prefix);

    ///
    /// \brief return a new stability center.
    ///
    const point_t& search(bundle_t&, scalar_t miu, tensor_size_t max_evals, scalar_t epsilon, const logger_t&);

private:
    // attributes
    const function_t& m_function;      ///<
    scalar_t          m_m1{0.5};       ///<
    scalar_t          m_m2{0.9};       ///<
    scalar_t          m_m3{1.0};       ///<
    scalar_t          m_m4{1.0};       ///<
    scalar_t          m_interpol{0.3}; ///< interpolation factor [tL, tR]: t = (1 - factor) * tL + factor * tR, see (2)
    scalar_t          m_extrapol{5.0}; ///< extrapolation factor [tR, +inf]: t = factor * tR, see (2)
    point_t           m_point;         ///<
};
} // namespace nano
