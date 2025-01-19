#include <solver/quasi.h>

using namespace nano;

namespace
{
template <class tvector>
void SR1(matrix_t& H, const tvector& dx, const tvector& dg)
{
    H = H + (dx - H * dg) * (dx - H * dg).transpose() / (dx - H * dg).dot(dg);
}

template <class tvector>
void SR1(matrix_t& H, const tvector& dx, const tvector& dg, const scalar_t r)
{
    const auto denom = (dx - H * dg).dot(dg);
    const auto apply = std::fabs(denom) >= r * dx.norm() * (dx - H * dg).norm();

    if (apply)
    {
        SR1(H, dx, dg);
    }
}

template <class tvector>
auto DFP_(const matrix_t& H, const tvector& dx, const tvector& dg)
{
    return H + (dx * dx.transpose()) / dx.dot(dg) - (H * dg * dg.transpose() * H) / (dg.transpose() * H * dg);
}

template <class tvector>
void DFP(matrix_t& H, const tvector& dx, const tvector& dg)
{
    H = DFP_(H, dx, dg);
}

template <class tvector>
auto BFGS_(const matrix_t& H, const tvector& dx, const tvector& dg)
{
    const auto I = matrix_t::identity(H.rows(), H.cols());

    return (I - dx * dg.transpose() / dx.dot(dg)) * H * (I - dg * dx.transpose() / dx.dot(dg)) +
           dx * dx.transpose() / dx.dot(dg);
}

template <class tvector>
void BFGS(matrix_t& H, const tvector& dx, const tvector& dg)
{
    H = BFGS_(H, dx, dg);
}

template <class tvector>
void HOSHINO(matrix_t& H, const tvector& dx, const tvector& dg)
{
    const auto phi = dx.dot(dg) / (dx.dot(dg) + dg.transpose() * H * dg);

    H = (1 - phi) * DFP_(H, dx, dg) + phi * BFGS_(H, dx, dg);
}

template <class tvector>
void FLETCHER(matrix_t& H, const tvector& dx, const tvector& dg)
{
    const auto phi = dx.dot(dg) / (dx.dot(dg) - dg.transpose() * H * dg);

    if (phi < scalar_t(0))
    {
        DFP(H, dx, dg);
    }
    else if (phi > scalar_t(1))
    {
        BFGS(H, dx, dg);
    }
    else
    {
        SR1(H, dx, dg);
    }
}
} // namespace

solver_quasi_t::solver_quasi_t(string_t id)
    : solver_t(std::move(id))
{
    parameter("solver::tolerance") = std::make_tuple(1e-4, 9e-1);

    register_parameter(parameter_t::make_enum("solver::quasi::initialization", quasi_initialization::identity));
}

solver_state_t solver_quasi_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    solver_t::warn_nonsmooth(function, logger);
    solver_t::warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto init      = parameter("solver::quasi::initialization").value<quasi_initialization>();

    auto cstate = solver_state_t{function, x0}; // current state
    auto pstate  = cstate;     // previous state
    auto descent = vector_t{}; // descent direction
    auto lsearch = make_lsearch();

    // current approximation of the Hessian's inverse
    matrix_t H = matrix_t::identity(function.size(), function.size());

    bool first_iteration = true;
    while (function.fcalls() + function.gcalls() < max_evals)
    {
        // descent direction
        descent = -H.matrix() * cstate.gx().vector();

        // restart:
        //  - if not a descent direction
        if (!cstate.has_descent(descent))
        {
            descent = -cstate.gx();
            H       = matrix_t::identity(H.rows(), H.cols());
        }

        // line-search
        pstate               = cstate;
        const auto iter_ok   = lsearch.get(cstate, descent, logger);
        if (solver_t::done_gradient_test(cstate, iter_ok, logger))
        {
            break;
        }

        // initialize the Hessian's inverse
        if (first_iteration && init == quasi_initialization::scaled)
        {
            const auto dx = cstate.x() - pstate.x();
            const auto dg = cstate.gx() - pstate.gx();
            H             = matrix_t::identity(H.rows(), H.cols()) * dx.dot(dg) / dg.dot(dg);
        }
        first_iteration = false;

        // update approximation of the Hessian
        update(pstate, cstate, H);
    }

    return cstate.valid() ? cstate : pstate;
}

solver_quasi_dfp_t::solver_quasi_dfp_t()
    : solver_quasi_t("dfp")
{
}

solver_quasi_bfgs_t::solver_quasi_bfgs_t()
    : solver_quasi_t("bfgs")
{
}

solver_quasi_hoshino_t::solver_quasi_hoshino_t()
    : solver_quasi_t("hoshino")
{
}

solver_quasi_fletcher_t::solver_quasi_fletcher_t()
    : solver_quasi_t("fletcher")
{
}

solver_quasi_sr1_t::solver_quasi_sr1_t()
    : solver_quasi_t("sr1")
{
    register_parameter(parameter_t::make_scalar("solver::quasi::sr1::r", 0, LT, 1e-8, LT, 1));
}

rsolver_t solver_quasi_dfp_t::clone() const
{
    return std::make_unique<solver_quasi_dfp_t>(*this);
}

rsolver_t solver_quasi_bfgs_t::clone() const
{
    return std::make_unique<solver_quasi_bfgs_t>(*this);
}

rsolver_t solver_quasi_hoshino_t::clone() const
{
    return std::make_unique<solver_quasi_hoshino_t>(*this);
}

rsolver_t solver_quasi_fletcher_t::clone() const
{
    return std::make_unique<solver_quasi_fletcher_t>(*this);
}

rsolver_t solver_quasi_sr1_t::clone() const
{
    return std::make_unique<solver_quasi_sr1_t>(*this);
}

void solver_quasi_sr1_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    const auto r = parameter("solver::quasi::sr1::r").value<scalar_t>();

    ::SR1(H, curr.x() - prev.x(), curr.gx() - prev.gx(), r);
}

void solver_quasi_dfp_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    ::DFP(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}

void solver_quasi_bfgs_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    ::BFGS(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}

void solver_quasi_hoshino_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    ::HOSHINO(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}

void solver_quasi_fletcher_t::update(const solver_state_t& prev, const solver_state_t& curr, matrix_t& H) const
{
    ::FLETCHER(H, curr.x() - prev.x(), curr.gx() - prev.gx());
}
