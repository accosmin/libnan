#include <nano/function/penalty.h>
#include <nano/solver/augmented.h>

using namespace nano;

static auto make_ro1(const solver_state_t& state, const scalar_t ro_min = 1e-6, const scalar_t ro_max = 10.0)
{
    const auto& f = state.f;
    const auto& h = state.ceq;
    const auto& g = state.cineq;
    const auto  G = g.array().max(0.0).matrix();

    const auto ro = 2.0 * std::fabs(f) / std::max(h.dot(h) + G.dot(G), 1e-6);
    return std::clamp(ro, ro_min, ro_max);
}

static auto make_criterion(const solver_state_t& state, const vector_t& miu, const scalar_t ro)
{
    const auto hinf = state.ceq.lpNorm<Eigen::Infinity>();
    const auto Vinf = state.cineq.array().max(-miu.array() / ro).matrix().lpNorm<Eigen::Infinity>();
    return std::max(hinf, Vinf);
}

solver_augmented_lagrangian_t::solver_augmented_lagrangian_t()
    : solver_t("augmented-lagrangian")
{
    type(solver_type::constrained);

    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();
    static constexpr auto fmin = std::numeric_limits<scalar_t>::lowest();

    register_parameter(parameter_t::make_scalar("solver::augmented::tau", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::augmented::gamma", 1.0, LT, 10.0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::augmented::miu_max", 0.0, LT, +1e+20, LT, fmax));
    register_parameter(parameter_t::make_integer("solver::augmented::max_outer_iters", 10, LE, 20, LE, 100));
    register_parameter(
        parameter_t::make_scalar_pair("solver::augmented::lambda", fmin, LT, -1e+20, LT, +1e+20, LT, fmax));
}

rsolver_t solver_augmented_lagrangian_t::clone() const
{
    return std::make_unique<solver_augmented_lagrangian_t>(*this);
}

solver_state_t solver_augmented_lagrangian_t::do_minimize(const function_t& function, const vector_t& x0) const
{
    const auto epsilon                  = parameter("solver::epsilon").value<scalar_t>();
    const auto max_evals                = parameter("solver::max_evals").value<tensor_size_t>();
    const auto tau                      = parameter("solver::augmented::tau").value<scalar_t>();
    const auto gamma                    = parameter("solver::augmented::gamma").value<scalar_t>();
    const auto miu_max                  = parameter("solver::augmented::miu_max").value<scalar_t>();
    const auto [lambda_min, lambda_max] = parameter("solver::augmented::lambda").value_pair<scalar_t>();
    const auto max_outers               = parameter("solver::augmented::max_outer_iters").value<tensor_size_t>();

    auto bstate        = solver_state_t{function, x0};
    auto ro            = make_ro1(bstate);
    auto old_criterion = scalar_t{0.0};
    auto lambda        = vector_t{vector_t::Zero(bstate.ceq.size())};
    auto miu           = vector_t{vector_t::Zero(bstate.cineq.size())};

    auto augmented_lagrangian_function = augmented_lagrangian_function_t{function, lambda, miu};
    auto solver                        = make_solver(augmented_lagrangian_function, epsilon, max_evals);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        augmented_lagrangian_function.penalty(ro);

        const auto cstate    = solver->minimize(augmented_lagrangian_function, x0);
        const auto criterion = make_criterion(cstate, miu, ro);
        const auto iter_ok   = cstate.valid();
        const auto converged = iter_ok && criterion < epsilon;

        // NB: the original function value should be returned!
        bstate.update(cstate.x);
        bstate.status = cstate.status;
        bstate.inner_iters += cstate.inner_iters;
        bstate.outer_iters++;

        if (done(function, bstate, iter_ok, converged))
        {
            break;
        }

        // update lagrange multipliers
        lambda.array() = (lambda.array() + ro * cstate.ceq.array()).max(lambda_min).min(lambda_max);
        miu.array()    = (miu.array() + ro * cstate.cineq.array()).max(0.0).min(miu_max);

        // update penalty parameter
        if (outer > 0 && criterion > tau * old_criterion)
        {
            ro = gamma * ro;
        }
        old_criterion = criterion;
    }

    return bstate;
}
