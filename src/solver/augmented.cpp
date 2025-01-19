#include <nano/function/penalty.h>
#include <solver/augmented.h>

using namespace nano;

namespace
{
auto make_ro1(const solver_state_t& state, const scalar_t ro_min = 1e-6, const scalar_t ro_max = 10.0)
{
    const auto  f = state.fx();
    const auto& h = state.ceq();
    const auto& g = state.cineq();
    const auto  G = g.array().max(0.0).matrix();

    const auto ro = 2.0 * std::fabs(f) / std::max(h.dot(h) + G.dot(G), 1e-6);
    return std::clamp(ro, ro_min, ro_max);
}

auto make_criterion(const solver_state_t& state, const vector_t& miu, const scalar_t ro)
{
    const auto hinf = state.ceq().lpNorm<Eigen::Infinity>();
    const auto Vinf = state.cineq().array().max(-miu.array() / ro).matrix().lpNorm<Eigen::Infinity>();
    return std::max(hinf, Vinf);
}
} // namespace

solver_augmented_lagrangian_t::solver_augmented_lagrangian_t()
    : solver_t("augmented-lagrangian")
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();
    static constexpr auto fmin = std::numeric_limits<scalar_t>::lowest();

    register_parameter(parameter_t::make_scalar("solver::augmented::epsilon0", 0.0, LT, 1e-8, LE, 1e-2));
    register_parameter(parameter_t::make_scalar("solver::augmented::epsilonK", 0.0, LT, 0.50, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::augmented::tau", 0.0, LT, 0.5, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::augmented::gamma", 1.0, LT, 10.0, LT, fmax));
    register_parameter(parameter_t::make_scalar("solver::augmented::miu_max", 0.0, LT, 1e+20, LT, fmax));
    register_parameter(parameter_t::make_integer("solver::augmented::max_outer_iters", 10, LE, 100, LE, 1000));
    register_parameter(
        parameter_t::make_scalar_pair("solver::augmented::lambda", fmin, LT, -1e+20, LT, +1e+20, LT, fmax));
}

rsolver_t solver_augmented_lagrangian_t::clone() const
{
    return std::make_unique<solver_augmented_lagrangian_t>(*this);
}

solver_state_t solver_augmented_lagrangian_t::do_minimize(const function_t& function, const vector_t& x0,
                                                          const logger_t& logger) const
{
    const auto max_evals                = parameter("solver::max_evals").value<tensor_size_t>();
    const auto epsilon0                 = parameter("solver::augmented::epsilon0").value<scalar_t>();
    const auto epsilonK                 = parameter("solver::augmented::epsilonK").value<scalar_t>();
    const auto tau                      = parameter("solver::augmented::tau").value<scalar_t>();
    const auto gamma                    = parameter("solver::augmented::gamma").value<scalar_t>();
    const auto miu_max                  = parameter("solver::augmented::miu_max").value<scalar_t>();
    const auto [lambda_min, lambda_max] = parameter("solver::augmented::lambda").value_pair<scalar_t>();
    const auto max_outers               = parameter("solver::augmented::max_outer_iters").value<tensor_size_t>();

    auto bstate        = solver_state_t{function, x0}; ///< best state
    auto cstate        = bstate;                       ///< current state
    auto ro            = make_ro1(bstate);
    auto lambda        = make_full_vector<scalar_t>(bstate.ceq().size(), 0.0);
    auto miu           = make_full_vector<scalar_t>(bstate.cineq().size(), 0.0);
    auto old_criterion = make_criterion(bstate, miu, ro);

    auto penalty_function = augmented_lagrangian_function_t{function, lambda, miu};
    auto solver           = make_solver(penalty_function, epsilon0, max_evals);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        // solve augmented lagrangian problem
        penalty_function.penalty(ro);
        const auto pstate = solver->minimize(penalty_function, bstate.x(), logger);
        cstate.update(pstate.x(), pstate.gx(), pstate.fx());

        // check convergence
        const auto iter_ok = cstate.valid();
        if (done_kkt_optimality_test(cstate, iter_ok, logger))
        {
            break;
        }

        const auto criterion = make_criterion(cstate, miu, ro);
        if (criterion < old_criterion)
        {
            bstate.update(cstate.x(), lambda, miu);
            solver->more_precise(epsilonK);
        }

        // update penalty parameter
        const auto old_ro = ro;
        if (outer > 0 && criterion > tau * old_criterion)
        {
            ro = gamma * ro;
        }
        old_criterion = criterion;

        // update lagrange multipliers
        lambda.array() = (lambda.array() + old_ro * cstate.ceq().array()).max(lambda_min).min(lambda_max);
        miu.array()    = (miu.array() + old_ro * cstate.cineq().array()).max(0.0).min(miu_max);
    }

    return bstate;
}
