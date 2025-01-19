#include <solver/cgd.h>

using namespace nano;

namespace
{
scalar_t HS(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    return cg.dot(cg - pg) / pd.dot(cg - pg);
}

scalar_t FR(const vector_t& pg, const vector_t&, const vector_t& cg)
{
    return cg.squaredNorm() / pg.squaredNorm();
}

scalar_t PR(const vector_t& pg, const vector_t&, const vector_t& cg)
{
    return cg.dot(cg - pg) / pg.squaredNorm();
}

scalar_t CD(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    return -cg.squaredNorm() / pd.dot(pg);
}

scalar_t LS(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    return -cg.dot(cg - pg) / pd.dot(pg);
}

scalar_t DY(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    return cg.squaredNorm() / pd.dot(cg - pg);
}

scalar_t N(const vector_t& pg, const vector_t& pd, const vector_t& cg, scalar_t eta) // N(+) - see (3)
{
    const auto y   = cg - pg;
    const auto div = +1 / pd.dot(y);

    const auto pd2 = pd.lpNorm<2>();
    const auto pg2 = pg.lpNorm<2>();
    eta            = -1 / (pd2 * std::min(eta, pg2));

    return std::max(eta, div * (y - 2 * pd * y.squaredNorm() * div).dot(cg.vector()));
}

scalar_t DYHS(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    return std::max(scalar_t(0), std::min(DY(pg, pd, cg), HS(pg, pd, cg)));
}

scalar_t DYCD(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    return cg.squaredNorm() / std::max(pd.dot(cg - pg), -pd.dot(pg));
}

scalar_t FRPR(const vector_t& pg, const vector_t& pd, const vector_t& cg)
{
    const auto fr = ::FR(pg, pd, cg);
    const auto pr = ::PR(pg, pd, cg);

    return (pr < -fr) ? -fr : (std::fabs(pr) <= fr) ? pr : fr;
}
} // namespace

solver_cgd_t::solver_cgd_t(string_t id)
    : solver_t(std::move(id))
{
    lsearchk("cgdescent");
    parameter("solver::tolerance") = std::make_tuple(1e-4, 1e-1);

    register_parameter(parameter_t::make_scalar("solver::cgd::orthotest", 0, LT, 0.1, LT, 1));
}

solver_state_t solver_cgd_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonsmooth(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto orthotest = parameter("solver::cgd::orthotest").value<scalar_t>();

    auto cstate   = solver_state_t{function, x0}; // current state
    auto pstate   = cstate;                       // previous state
    auto lsearch  = make_lsearch();
    auto cdescent = vector_t{}; // current descent direction
    auto pdescent = vector_t{}; // previous descent direction

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // descent direction
        if (cdescent.size() == 0)
        {
            cdescent = -cstate.gx();
        }
        else
        {
            const auto beta = this->beta(pstate.gx(), pdescent, cstate.gx());
            cdescent        = -cstate.gx() + beta * pdescent;

            // restart:
            //  - if not a descent direction
            //  - or two consecutive gradients far from being orthogonal
            //      (see "Numerical optimization", Nocedal & Wright, 2nd edition, p.124-125)
            if (!cstate.has_descent(cdescent) ||
                (std::fabs(cstate.gx().dot(pstate.gx())) >= orthotest * cstate.gx().dot(cstate.gx())))
            {
                cdescent = -cstate.gx();
            }
        }

        pstate   = cstate;
        pdescent = cdescent;

        // line-search
        const auto iter_ok = lsearch.get(cstate, cdescent, logger);
        if (solver_t::done_gradient_test(cstate, iter_ok, logger))
        {
            break;
        }
    }

    return cstate.valid() ? cstate : pstate;
}

solver_cgd_cd_t::solver_cgd_cd_t()
    : solver_cgd_t("cgd-cd")
{
}

solver_cgd_dy_t::solver_cgd_dy_t()
    : solver_cgd_t("cgd-dy")
{
}

solver_cgd_fr_t::solver_cgd_fr_t()
    : solver_cgd_t("cgd-fr")
{
}

solver_cgd_hs_t::solver_cgd_hs_t()
    : solver_cgd_t("cgd-hs")
{
}

solver_cgd_ls_t::solver_cgd_ls_t()
    : solver_cgd_t("cgd-ls")
{
}

solver_cgd_pr_t::solver_cgd_pr_t()
    : solver_cgd_t("cgd-pr")
{
}

solver_cgd_dycd_t::solver_cgd_dycd_t()
    : solver_cgd_t("cgd-dycd")
{
}

solver_cgd_dyhs_t::solver_cgd_dyhs_t()
    : solver_cgd_t("cgd-dyhs")
{
}

solver_cgd_frpr_t::solver_cgd_frpr_t()
    : solver_cgd_t("cgd-frpr")
{
}

solver_cgd_n_t::solver_cgd_n_t()
    : solver_cgd_t("cgd-n")
{
    register_parameter(parameter_t::make_scalar("solver::cgdN::eta", 0, LT, 0.01, LT, 1e+6));
}

rsolver_t solver_cgd_cd_t::clone() const
{
    return std::make_unique<solver_cgd_cd_t>(*this);
}

rsolver_t solver_cgd_dy_t::clone() const
{
    return std::make_unique<solver_cgd_dy_t>(*this);
}

rsolver_t solver_cgd_fr_t::clone() const
{
    return std::make_unique<solver_cgd_fr_t>(*this);
}

rsolver_t solver_cgd_hs_t::clone() const
{
    return std::make_unique<solver_cgd_hs_t>(*this);
}

rsolver_t solver_cgd_ls_t::clone() const
{
    return std::make_unique<solver_cgd_ls_t>(*this);
}

rsolver_t solver_cgd_pr_t::clone() const
{
    return std::make_unique<solver_cgd_pr_t>(*this);
}

rsolver_t solver_cgd_dycd_t::clone() const
{
    return std::make_unique<solver_cgd_dycd_t>(*this);
}

rsolver_t solver_cgd_dyhs_t::clone() const
{
    return std::make_unique<solver_cgd_dyhs_t>(*this);
}

rsolver_t solver_cgd_frpr_t::clone() const
{
    return std::make_unique<solver_cgd_frpr_t>(*this);
}

rsolver_t solver_cgd_n_t::clone() const
{
    return std::make_unique<solver_cgd_n_t>(*this);
}

scalar_t solver_cgd_hs_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    // HS+ - see (1)
    return std::max(::HS(pg, pd, cg), scalar_t(0));
}

scalar_t solver_cgd_fr_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    return ::FR(pg, pd, cg);
}

scalar_t solver_cgd_pr_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    // PR+ - see (1)
    return std::max(::PR(pg, pd, cg), scalar_t(0));
}

scalar_t solver_cgd_cd_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    return ::CD(pg, pd, cg);
}

scalar_t solver_cgd_ls_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    // LS+ - see (1)
    return std::max(::LS(pg, pd, cg), scalar_t(0));
}

scalar_t solver_cgd_dy_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    return ::DY(pg, pd, cg);
}

scalar_t solver_cgd_n_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    const auto eta = parameter("solver::cgdN::eta").value<scalar_t>();

    return ::N(pg, pd, cg, eta);
}

scalar_t solver_cgd_dycd_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    return ::DYCD(pg, pd, cg);
}

scalar_t solver_cgd_dyhs_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    return ::DYHS(pg, pd, cg);
}

scalar_t solver_cgd_frpr_t::beta(const vector_t& pg, const vector_t& pd, const vector_t& cg) const
{
    return ::FRPR(pg, pd, cg);
}
