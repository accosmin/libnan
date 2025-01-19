#include <nano/function/penalty.h>
#include <solver/penalty.h>

using namespace nano;

solver_penalty_t::solver_penalty_t(string_t id)
    : solver_t(std::move(id))
{
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilonK", 0.0, LT, 0.5, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::penalty::eta", 1.0, LT, 5.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::penalty0", 0.0, LT, 10.0, LE, 1e+3));
    register_parameter(parameter_t::make_integer("solver::penalty::max_outer_iters", 10, LE, 20, LE, 100));

    parameter("solver::patience") = 2;
}

solver_state_t solver_penalty_t::minimize(penalty_function_t& penalty_function, const vector_t& x0,
                                          const logger_t& logger) const
{
    const auto max_evals  = parameter("solver::max_evals").value<tensor_size_t>();
    const auto eta        = parameter("solver::penalty::eta").value<scalar_t>();
    const auto epsilon0   = parameter("solver::penalty::epsilon0").value<scalar_t>();
    const auto epsilonK   = parameter("solver::penalty::epsilonK").value<scalar_t>();
    const auto penalty0   = parameter("solver::penalty::penalty0").value<scalar_t>();
    const auto max_outers = parameter("solver::penalty::max_outer_iters").value<tensor_size_t>();

    auto penalty = penalty0;
    auto solver  = make_solver(penalty_function, epsilon0, max_evals);
    auto bstate  = solver_state_t{penalty_function.function(), x0};

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        // solve the penalty problem
        penalty_function.penalty(penalty);
        const auto cstate = solver->minimize(penalty_function, bstate.x(), logger);

        // increase penalty until the solution is bounded
        const auto iter_ok = cstate.valid();
        if (!iter_ok)
        {
            penalty *= eta;
            continue;
        }

        // check convergence
        bstate.update(cstate.x());
        if (done_value_test(bstate, iter_ok, logger))
        {
            break;
        }

        // update penalty parameter
        penalty *= eta;
        solver->more_precise(epsilonK);
    }

    return bstate;
}

solver_linear_penalty_t::solver_linear_penalty_t()
    : solver_penalty_t("linear-penalty")
{
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilon0", 0.0, LT, 1e-8, LE, 1e-2));
}

rsolver_t solver_linear_penalty_t::clone() const
{
    return std::make_unique<solver_linear_penalty_t>(*this);
}

solver_state_t solver_linear_penalty_t::do_minimize(const function_t& function, const vector_t& x0,
                                                    const logger_t& logger) const
{
    auto penalty_function = linear_penalty_function_t{function};

    return minimize(penalty_function, x0, logger);
}

solver_quadratic_penalty_t::solver_quadratic_penalty_t()
    : solver_penalty_t("quadratic-penalty")
{
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilon0", 0.0, LT, 1e-6, LE, 1e-2));
}

rsolver_t solver_quadratic_penalty_t::clone() const
{
    return std::make_unique<solver_quadratic_penalty_t>(*this);
}

solver_state_t solver_quadratic_penalty_t::do_minimize(const function_t& function, const vector_t& x0,
                                                       const logger_t& logger) const
{
    auto penalty_function = quadratic_penalty_function_t{function};

    return minimize(penalty_function, x0, logger);
}
