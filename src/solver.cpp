#include <mutex>
#include <nano/core/logger.h>
#include <nano/solver/cgd.h>
#include <nano/solver/ellipsoid.h>
#include <nano/solver/gd.h>
#include <nano/solver/lbfgs.h>
#include <nano/solver/osga.h>
#include <nano/solver/quasi.h>

using namespace nano;

solver_t::solver_t()
{
    lsearch0("quadratic");
    lsearchk("morethuente");

    register_parameter(parameter_t::make_scalar("solver::epsilon", 0, LT, 1e-6, LE, 1e-1));
    register_parameter(parameter_t::make_integer("solver::max_evals", 10, LE, 1000, LE, 1e+9));
    register_parameter(parameter_t::make_scalar_pair("solver::tolerance", 0, LT, 1e-4, LT, 0.1, LT, 1));
}

void solver_t::lsearch0(const string_t& id)
{
    lsearch0(id, lsearch0_t::all().get(id));
}

void solver_t::lsearch0(const string_t& id, rlsearch0_t&& init)
{
    critical(!init, "solver: invalid line-search initialization (", id, ")!");

    m_lsearch0_id = id;
    m_lsearch0    = std::move(init);
}

void solver_t::lsearchk(const string_t& id)
{
    lsearchk(id, lsearchk_t::all().get(id));
}

void solver_t::lsearchk(const string_t& id, rlsearchk_t&& strategy)
{
    critical(!strategy, "solver: invalid line-search strategy (", id, ")!");

    m_lsearchk_id = id;
    m_lsearchk    = std::move(strategy);
}

void solver_t::lsearch0_logger(const lsearch0_t::logger_t& logger)
{
    m_lsearch0->logger(logger);
}

void solver_t::lsearchk_logger(const lsearchk_t::logger_t& logger)
{
    m_lsearchk->logger(logger);
}

void solver_t::logger(const logger_t& logger)
{
    m_logger = logger;
}

void solver_t::monotonic(bool monotonic)
{
    m_monotonic = monotonic;
}

bool solver_t::monotonic() const
{
    return m_monotonic;
}

lsearch_t solver_t::make_lsearch() const
{
    // NB: create new line-search objects:
    //  - to have the solver thread-safe
    //  - to start with a fresh line-search history (needed for some strategies like CG_DESCENT)
    auto lsearch0 = m_lsearch0->clone();
    auto lsearchk = m_lsearchk->clone();

    lsearch0->parameter("lsearch0::epsilon")   = parameter("solver::epsilon").value<scalar_t>();
    lsearchk->parameter("lsearchk::tolerance") = parameter("solver::tolerance").value_pair<scalar_t>();
    return lsearch_t{std::move(lsearch0), std::move(lsearchk)};
}

solver_state_t solver_t::minimize(const function_t& function, const vector_t& x0) const
{
    critical(function.size() != x0.size(), "solver: incompatible initial point (", x0.size(),
             " dimensions), expecting ", function.size(), " dimensions!");

    function.clear_statistics();

    return do_minimize(function, x0);
}

bool solver_t::done(const function_t& function, solver_state_t& state, bool iter_ok, bool converged) const
{
    state.m_fcalls = function.fcalls();
    state.m_gcalls = function.gcalls();

    const auto step_ok = iter_ok && state;

    if (converged || !step_ok)
    {
        // either converged or failed
        state.m_status = converged ? solver_state_t::status::converged : solver_state_t::status::failed;
        log(state);
        return true;
    }
    else if (!log(state))
    {
        // stopping was requested
        state.m_status = solver_state_t::status::stopped;
        return true;
    }

    // OK, go on with the optimization
    return false;
}

bool solver_t::log(solver_state_t& state) const
{
    const auto status = !m_logger ? true : m_logger(state);
    state.m_iterations++;
    return status;
}

solver_factory_t& solver_t::all()
{
    static auto manager = solver_factory_t{};
    const auto  op      = []()
    {
        manager.add<solver_gd_t>("gd", "gradient descent");
        manager.add<solver_cgd_pr_t>("cgd", "conjugate gradient descent (default)");
        manager.add<solver_cgd_n_t>("cgd-n", "conjugate gradient descent (N+)");
        manager.add<solver_cgd_hs_t>("cgd-hs", "conjugate gradient descent (HS+)");
        manager.add<solver_cgd_fr_t>("cgd-fr", "conjugate gradient descent (FR)");
        manager.add<solver_cgd_pr_t>("cgd-pr", "conjugate gradient descent (PR+)");
        manager.add<solver_cgd_cd_t>("cgd-cd", "conjugate gradient descent (CD)");
        manager.add<solver_cgd_ls_t>("cgd-ls", "conjugate gradient descent (LS+)");
        manager.add<solver_cgd_dy_t>("cgd-dy", "conjugate gradient descent (DY)");
        manager.add<solver_cgd_dycd_t>("cgd-dycd", "conjugate gradient descent (DYCD)");
        manager.add<solver_cgd_dyhs_t>("cgd-dyhs", "conjugate gradient descent (DYHS)");
        manager.add<solver_cgd_frpr_t>("cgd-prfr", "conjugate gradient descent (FRPR)");
        manager.add<solver_osga_t>("osga", "optimal sub-gradient algorithm (OSGA)");
        manager.add<solver_lbfgs_t>("lbfgs", "limited-memory BFGS");
        manager.add<solver_quasi_dfp_t>("dfp", "quasi-newton method (DFP)");
        manager.add<solver_quasi_sr1_t>("sr1", "quasi-newton method (SR1)");
        manager.add<solver_quasi_bfgs_t>("bfgs", "quasi-newton method (BFGS)");
        manager.add<solver_quasi_hoshino_t>("hoshino", "quasi-newton method (Hoshino formula)");
        manager.add<solver_quasi_fletcher_t>("fletcher", "quasi-newton method (Fletcher's switch)");
        manager.add<solver_ellipsoid_t>("ellipsoid", "ellipsoid method");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
