#include <nano/solver/penalty.h>

using namespace nano;

static auto initial_params(const estimator_t& estimator)
{
    const auto eta1       = estimator.parameter("solver::penalty::eta1").template value<scalar_t>();
    const auto eta2       = estimator.parameter("solver::penalty::eta2").template value<scalar_t>();
    const auto cutoff     = estimator.parameter("solver::penalty::cutoff").template value<scalar_t>();
    const auto epsilon    = estimator.parameter("solver::penalty::epsilon").template value<scalar_t>();
    const auto penalty0   = estimator.parameter("solver::penalty::penalty0").template value<scalar_t>();
    const auto max_outers = estimator.parameter("solver::penalty::max_outer_iters").template value<tensor_size_t>();

    return std::make_tuple(eta1, eta2, cutoff, epsilon, penalty0, max_outers);
}

template <typename tpenalty>
static auto make_penalty_function(const function_t& function, const scalar_t cutoff)
{
    auto penalty_function = tpenalty{function};
    penalty_function.cutoff(cutoff);
    return penalty_function;
}

solver_penalty_t::solver_penalty_t()
{
    register_parameter(parameter_t::make_scalar("solver::penalty::eta1", 0.0, LT, 0.1, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::penalty::eta2", 1.0, LT, 20.0, LE, 1e+3));
    register_parameter(parameter_t::make_scalar("solver::penalty::cutoff", 0, LT, 1e+3, LE, 1e+10));
    register_parameter(parameter_t::make_scalar("solver::penalty::epsilon", 0, LT, 1e-8, LE, 1e-1));
    register_parameter(parameter_t::make_scalar("solver::penalty::penalty0", 0.0, LT, 1.0, LE, 1e+3));
    register_parameter(parameter_t::make_integer("solver::penalty::max_outer_iters", 10, LE, 20, LE, 100));
}

solver_penalty_t::~solver_penalty_t() = default;

void solver_penalty_t::logger(const solver_t::logger_t& logger)
{
    m_logger = logger;
}

bool solver_penalty_t::done(const solver_state_t& curr_state, solver_state_t& best_state, scalar_t epsilon) const
{
    best_state.f = curr_state.f;
    best_state.x = curr_state.x;
    best_state.g = curr_state.g;
    best_state.p = curr_state.p;
    best_state.fcalls += curr_state.fcalls;
    best_state.gcalls += curr_state.gcalls;
    best_state.inner_iters += curr_state.inner_iters;
    best_state.outer_iters++;

    auto done = false;
    if (best_state.p.sum() < epsilon)
    {
        done              = true;
        best_state.status = solver_status::converged;
    }

    if (m_logger && !m_logger(best_state))
    {
        done = true;
        if (best_state.status != solver_status::converged)
        {
            best_state.status = solver_status::stopped;
        }
    }

    return done;
}

solver_state_t solver_linear_penalty_t::minimize(const solver_t& solver, const function_t& function,
                                                 const vector_t& x0) const
{
    [[maybe_unused]] const auto [eta1, eta2, cutoff, epsilon, penalty0, max_outers] = initial_params(*this);

    auto penalty          = penalty0;
    auto best_state       = solver_state_t{function, x0};
    auto penalty_function = make_penalty_function<linear_penalty_function_t>(function, cutoff);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);

        const auto curr_state = solver.minimize(penalty_function, best_state.x);
        if (done(curr_state, best_state, epsilon))
        {
            break;
        }

        penalty *= eta2;
    }

    return best_state;
}

solver_state_t solver_quadratic_penalty_t::minimize(const solver_t& solver, const function_t& function,
                                                    const vector_t& x0) const
{
    [[maybe_unused]] const auto [eta1, eta2, cutoff, epsilon, penalty0, max_outers] = initial_params(*this);

    auto penalty          = penalty0;
    auto best_state       = solver_state_t{function, x0};
    auto penalty_function = make_penalty_function<quadratic_penalty_function_t>(function, cutoff);

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);

        const auto curr_state = solver.minimize(penalty_function, best_state.x);
        if (done(curr_state, best_state, epsilon))
        {
            break;
        }

        penalty *= eta2;
    }

    return best_state;
}

solver_state_t solver_linear_quadratic_penalty_t::minimize(const solver_t& solver, const function_t& function,
                                                           const vector_t& x0) const
{
    [[maybe_unused]] const auto [eta1, eta2, cutoff, epsilon, penalty0, max_outers] = initial_params(*this);

    auto penalty          = penalty0;
    auto best_state       = solver_state_t{function, x0};
    auto penalty_function = make_penalty_function<linear_quadratic_penalty_function_t>(function, cutoff);

    auto smoothing = best_state.p.maxCoeff();

    for (tensor_size_t outer = 0; outer < max_outers; ++outer)
    {
        penalty_function.penalty(penalty);
        penalty_function.smoothing(smoothing);

        const auto curr_state = solver.minimize(penalty_function, best_state.x);
        if (done(curr_state, best_state, epsilon))
        {
            break;
        }

        if (curr_state.p.maxCoeff() <= smoothing)
        {
            smoothing = eta1 * curr_state.p.maxCoeff();
        }
        else
        {
            penalty *= eta2;
        }
    }

    return best_state;
}
