#include <deque>
#include <solver/lbfgs.h>

using namespace nano;

solver_lbfgs_t::solver_lbfgs_t()
    : solver_t("lbfgs")
{
    lsearchk("cgdescent");
    parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);

    register_parameter(parameter_t::make_integer("solver::lbfgs::history", 1, LE, 20, LE, 1000));
}

rsolver_t solver_lbfgs_t::clone() const
{
    return std::make_unique<solver_lbfgs_t>(*this);
}

solver_state_t solver_lbfgs_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonsmooth(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto history   = parameter("solver::lbfgs::history").value<size_t>();

    auto cstate  = solver_state_t{function, x0}; // current state
    auto pstate  = cstate;                       // previous state
    auto lsearch = make_lsearch();

    vector_t             q, r;
    std::deque<vector_t> ss, ys;
    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // descent direction
        //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.178)
        q = cstate.gx();

        const auto hsize = ss.size();

        std::vector<scalar_t> alphas(hsize);
        for (size_t j = 0; j < hsize; ++j)
        {
            const auto& s = ss[hsize - 1 - j];
            const auto& y = ys[hsize - 1 - j];

            const scalar_t alpha = s.dot(q) / s.dot(y);
            q -= alpha * y;
            alphas[j] = alpha;
        }

        if (ss.empty())
        {
            r = q;
        }
        else
        {
            const auto& s = ss[hsize - 1];
            const auto& y = ys[hsize - 1];

            r = s.dot(y) / y.dot(y) * q;
        }

        for (size_t j = 0; j < hsize; ++j)
        {
            const auto& s = ss[j];
            const auto& y = ys[j];

            const scalar_t alpha = alphas[hsize - 1 - j];
            const scalar_t beta  = y.dot(r) / s.dot(y);
            r += s * (alpha - beta);
        }

        auto& descent = r;
        descent       = -r;

        // force descent direction
        const auto has_descent = cstate.has_descent(descent);
        if (!has_descent)
        {
            descent = -cstate.gx();
        }

        // line-search
        pstate               = cstate;
        const auto iter_ok   = lsearch.get(cstate, descent, logger);
        if (solver_t::done_gradient_test(cstate, iter_ok, logger))
        {
            break;
        }

        // Skip the update if the curvature condition is not satisfied
        //      "A Multi-Batch L-BFGS Method for Machine Learning", page 6 - the non-convex case
        if (has_descent)
        {
            ss.emplace_back(cstate.x() - pstate.x());
            ys.emplace_back(cstate.gx() - pstate.gx());
            if (ss.size() > history)
            {
                ss.pop_front();
                ys.pop_front();
            }
        }
        else
        {
            ss.clear();
            ys.clear();
        }
    }

    return cstate.valid() ? cstate : pstate;
}
