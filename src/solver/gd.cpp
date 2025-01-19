#include <solver/gd.h>

using namespace nano;

solver_gd_t::solver_gd_t()
    : solver_t("gd")
{
    parameter("solver::tolerance") = std::make_tuple(1e-1, 9e-1);
}

rsolver_t solver_gd_t::clone() const
{
    return std::make_unique<solver_gd_t>(*this);
}

solver_state_t solver_gd_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonsmooth(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();

    auto state = solver_state_t{function, x0};
    auto lsearch = make_lsearch();
    auto descent = vector_t{function.size()};

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        descent = -state.gx();

        // line-search
        const auto iter_ok = lsearch.get(state, descent, logger);
        if (solver_t::done_gradient_test(state, iter_ok, logger))
        {
            break;
        }
    }

    return state;
} // LCOV_EXCL_LINE
