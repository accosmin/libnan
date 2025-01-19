#include <lsearch0/cgdescent.h>

using namespace nano;

lsearch0_cgdescent_t::lsearch0_cgdescent_t()
    : lsearch0_t("cgdescent")
{
    register_parameter(parameter_t::make_scalar("lsearch0::cgdescent::phi0", 0, LT, 0.01, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearch0::cgdescent::phi1", 0, LT, 0.10, LT, 1));
    register_parameter(parameter_t::make_scalar("lsearch0::cgdescent::phi2", 1, LT, 2.00, LT, 1e+6));
}

rlsearch0_t lsearch0_cgdescent_t::clone() const
{
    return std::make_unique<lsearch0_cgdescent_t>(*this);
}

scalar_t lsearch0_cgdescent_t::get(const solver_state_t& state, const vector_t& descent, const scalar_t last_step_size)
{
    const auto phi0 = parameter("lsearch0::cgdescent::phi0").value<scalar_t>();
    const auto phi1 = parameter("lsearch0::cgdescent::phi1").value<scalar_t>();
    const auto phi2 = parameter("lsearch0::cgdescent::phi2").value<scalar_t>();

    scalar_t t0 = 0;
    if (last_step_size < 0.0)
    {
        const auto xnorm = state.x().lpNorm<Eigen::Infinity>();
        const auto fnorm = std::fabs(state.fx());

        if (xnorm > 0)
        {
            t0 = phi0 * xnorm / state.gx().lpNorm<Eigen::Infinity>();
        }
        else if (fnorm > 0)
        {
            t0 = phi0 * fnorm / state.gx().squaredNorm();
        }
        else
        {
            t0 = 1;
        }
    }
    else
    {
        const auto& funct = state.function();
        const auto  prevt = last_step_size;
        const auto  step0 = lsearch_step_t{0, state.fx(), state.dg(descent)};
        const auto  trial = vector_t{state.x() + prevt * phi1 * descent};
        const auto  stepx = lsearch_step_t{prevt * phi1, funct(trial), 0};

        bool       convexity = false;
        const auto tq        = lsearch_step_t::quadratic(step0, stepx, &convexity);
        if (stepx.f < step0.f && convexity)
        {
            t0 = tq;
        }
        else
        {
            t0 = last_step_size * phi2;
        }
    }

    return t0;
}
