#include <Eigen/Dense>
#include <nano/program/solver.h>
#include <nano/program/util.h>

using namespace nano;
using namespace nano::program;

namespace
{
template <class tprogram>
vector_t make_x0(const tprogram& program)
{
    const auto x0 = program.make_strictly_feasible();
    if (!x0)
    {
        return vector_t::zero(program.m_c.size());
    }
    else
    {
        return x0.value();
    }
}

auto make_smax(const vector_t& u, const vector_t& du)
{
    assert(u.size() == du.size());

    auto smax = std::numeric_limits<scalar_t>::max();
    for (tensor_size_t i = 0, size = u.size(); i < size; ++i)
    {
        if (du(i) < 0.0)
        {
            smax = std::min(smax, -u(i) / du(i));
        }
    }

    return std::min(smax, 1.0);
}

scalar_t normalize(matrix_t& A, vector_t& b, const scalar_t min_norm = 1e-3)
{
    const auto denom = std::max({min_norm, A.lpNorm<2>(), b.lpNorm<2>()});
    A.array() /= denom;
    b.array() /= denom;
    return denom;
}

struct reducer_t
{
    reducer_t() = default;

    reducer_t(matrix_t& A, vector_t& b) { reduce(A, b); }
};
} // namespace

struct solver_t::program_t
{
    explicit program_t(const linear_program_t& program)
        : program_t(matrix_t{}, program.m_c, program.m_eq.m_A, program.m_eq.m_b, program.m_ineq.m_A, program.m_ineq.m_b)
    {
    }

    explicit program_t(const quadratic_program_t& program)
        : program_t(program.m_Q, program.m_c, program.m_eq.m_A, program.m_eq.m_b, program.m_ineq.m_A,
                    program.m_ineq.m_b)
    {
    }

    program_t(matrix_t Q, vector_t c, matrix_t A, vector_t b, matrix_t G, vector_t h)
        : m_Q(std::move(Q))
        , m_c(std::move(c))
        , m_A(std::move(A))
        , m_b(std::move(b))
        , m_G(std::move(G))
        , m_h(std::move(h))
        , m_reducer(m_A, m_b)           // remove linear dependant linear constraints (if any)
        , m_mufx(::normalize(m_Q, m_c)) // normalize objective
        , m_lmat(n() + p(), n() + p())
        , m_lvec(n() + p())
        , m_lsol(n() + p())
    {
        // normalize constraints
        ::normalize(m_A, m_b);
        ::normalize(m_G, m_h);

        // allocate buffers for the linear system of equations
        const auto n = this->n();
        const auto p = this->p();

        if (p > 0)
        {
            m_lmat.block(0, n, n, p) = m_A.transpose();
            m_lmat.block(n, 0, p, n) = m_A.matrix();
        }
        m_lmat.block(n, n, p, p).array() = 0.0;
    }

    tensor_size_t n() const { return m_c.size(); }

    tensor_size_t p() const { return m_A.rows(); }

    tensor_size_t m() const { return m_G.rows(); }

    bool feasible(const solver_state_t& state) const
    {
        const auto& A = m_A;
        const auto& b = m_b;
        const auto& G = m_G;
        const auto& h = m_h;

        return (A.rows() == 0 || (A * state.m_x - b).lpNorm<2>() < epsilon2<scalar_t>()) &&
               (G.rows() == 0 || (G * state.m_x - h).maxCoeff() < epsilon2<scalar_t>());
    }

    const matrix_t& Q() const
    {
        assert(m_Q.size() > 0);
        return m_Q;
    }

    template <class thessvar, class trdual, class trprim>
    const vector_t& solve(const thessvar& hessvar, const trdual& rdual, const trprim& rprim) const
    {
        const auto n = this->n();
        const auto p = this->p();

        // setup additional hessian components
        if (!m_Q.size())
        {
            m_lmat.block(0, 0, n, n) = -hessvar;
        }
        else
        {
            m_lmat.block(0, 0, n, n) = Q() - hessvar;
        }

        // setup residuals
        m_lvec.segment(0, n) = -rdual;
        m_lvec.segment(n, p) = -rprim;

        // solve the system
        m_ldlt.compute(m_lmat.matrix());
        m_lsol.vector() = m_ldlt.solve(m_lvec.vector());
        return m_lsol;
    }

    template <class tvector>
    void update(const tvector& x, const tvector& u, const tvector& v, const scalar_t miu, solver_state_t& state) const
    {
        const auto m = this->m();
        const auto p = this->p();

        // objective
        if (!m_Q.size())
        {
            state.m_fx    = x.dot(m_c.vector());
            state.m_rdual = m_c;
        }
        else
        {
            state.m_fx    = 0.5 * x.dot(Q() * x) + x.dot(m_c.vector());
            state.m_rdual = Q() * x + m_c;
        }
        state.m_fx *= m_mufx; // NB: rescale the objective!

        // surrogate duality gap
        if (m > 0)
        {
            state.m_eta = -u.dot(m_G * x - m_h);
        }

        // residual contributions of linear equality constraints
        if (p > 0)
        {
            state.m_rdual += m_A.transpose() * v;
            state.m_rprim = m_A * x - m_b;
        }

        // residual contributions of linear inequality constraints
        if (m > 0)
        {
            const auto sm = static_cast<scalar_t>(m);
            state.m_rdual += m_G.transpose() * u;
            state.m_rcent = -state.m_eta / (miu * sm) - u.array() * (m_G * x - m_h).array();
        }
    }

    using lin_solver_t = Eigen::LDLT<eigen_matrix_t<scalar_t>>;

    // attributes
    matrix_t             m_Q;         ///< objective: 1/2 * x.dot(Q * x) + c.dot(x)
    vector_t             m_c;         ///<
    matrix_t             m_A;         ///< equality constraint: A * x = b
    vector_t             m_b;         ///<
    matrix_t             m_G;         ///< inequality constraint: Gx <= h
    vector_t             m_h;         ///<
    reducer_t            m_reducer;   ///<
    scalar_t             m_mufx{1.0}; ///< scaling coefficient of the objective
    mutable lin_solver_t m_ldlt;      ///< buffers for the linear system of equations coupling (dx, dv)
    mutable matrix_t     m_lmat;      ///<
    mutable vector_t     m_lvec;      ///<
    mutable vector_t     m_lsol;      ///<
};

solver_t::solver_t()
{
    register_parameter(parameter_t::make_scalar("solver::s0", 0.0, LT, 0.999, LE, 1.0));
    register_parameter(parameter_t::make_scalar("solver::miu", 1.0, LT, 10.0, LE, 1e+6));
    register_parameter(parameter_t::make_scalar("solver::alpha", 0.0, LT, 1e-2, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::beta", 0.0, LT, 0.9, LT, 1.0));
    register_parameter(parameter_t::make_scalar("solver::epsilon", 0.0, LE, 1e-10, LE, 1e-3));
    register_parameter(parameter_t::make_scalar("solver::epsilon0", 0.0, LE, 1e-32, LE, 1e-3));
    register_parameter(parameter_t::make_integer("solver::max_iters", 10, LE, 300, LE, 1000));
    register_parameter(parameter_t::make_integer("solver::max_lsearch_iters", 10, LE, 50, LE, 1000));
}

solver_state_t solver_t::solve(const linear_program_t& program, const logger_t& logger) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program}, logger)
                                   : solve_with_inequality(program_t{program}, make_x0(program), logger);
}

solver_state_t solver_t::solve(const quadratic_program_t& program, const logger_t& logger) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program}, logger)
                                   : solve_with_inequality(program_t{program}, make_x0(program), logger);
}

solver_state_t solver_t::solve(const linear_program_t& program, const vector_t& x0, const logger_t& logger) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program}, logger)
                                   : solve_with_inequality(program_t{program}, x0, logger);
}

solver_state_t solver_t::solve(const quadratic_program_t& program, const vector_t& x0, const logger_t& logger) const
{
    return !program.m_ineq.valid() ? solve_without_inequality(program_t{program}, logger)
                                   : solve_with_inequality(program_t{program}, x0, logger);
}

solver_state_t solver_t::solve_with_inequality(const program_t& program, const vector_t& x0,
                                               const logger_t& logger) const
{
    const auto s0                = parameter("solver::s0").value<scalar_t>();
    const auto miu               = parameter("solver::miu").value<scalar_t>();
    const auto alpha             = parameter("solver::alpha").value<scalar_t>();
    const auto beta              = parameter("solver::beta").value<scalar_t>();
    const auto epsilon           = parameter("solver::epsilon").value<scalar_t>();
    const auto epsilon0          = parameter("solver::epsilon0").value<scalar_t>();
    const auto max_iters         = parameter("solver::max_iters").value<tensor_size_t>();
    const auto max_lsearch_iters = parameter("solver::max_lsearch_iters").value<tensor_size_t>();

    const auto& G = program.m_G;
    const auto& h = program.m_h;
    const auto  n = program.n();
    const auto  p = program.p();
    const auto  m = program.m();

    auto state = solver_state_t{n, m, p};
    state.m_x  = x0;

    // the starting point must be strictly feasible wrt inequality constraints
    if (const auto mGxh = (G * x0 - h).maxCoeff(); mGxh >= 0.0)
    {
        state.m_status = solver_status::unfeasible;
        logger.info("[program]: ", state, ".\n");
        return state;
    }

    auto dx = vector_t{n};
    auto du = vector_t{m};
    auto dv = vector_t{p};

    // current solver state
    state.m_u = -1.0 / (G * x0 - h).array();
    state.m_v = vector_t::zero(p);

    // update residuals
    program.update(state.m_x, state.m_u, state.m_v, miu, state);

    // primal-dual interior-point solver...
    for (state.m_iters = 0; state.m_iters < max_iters; ++state.m_iters)
    {
        const auto prev_eta   = state.m_eta;
        const auto prev_rdual = state.m_rdual.lpNorm<2>();
        const auto prev_rprim = state.m_rprim.lpNorm<2>();

        // solve primal-dual linear system of equations to get (dx, du, dv)
        const auto Gxh = G * state.m_x - h;
        program.solve(G.transpose() * (state.m_u.array() / Gxh.array()).matrix().asDiagonal() * G.matrix(),
                      state.m_rdual + G.transpose() * (state.m_rcent.array() / Gxh.array()).matrix(), state.m_rprim);

        dx = program.m_lsol.segment(0, n);
        dv = program.m_lsol.segment(n, p);
        du = (state.m_rcent.array() - state.m_u.array() * (G * dx).array()) / Gxh.array();

        state.m_ldlt_rcond    = program.m_ldlt.rcond();
        state.m_ldlt_positive = program.m_ldlt.isPositive();

        // stop if the linear system of equations is not stable
        if (!std::isfinite(state.m_ldlt_rcond) || !dx.all_finite() || !dv.all_finite() || !du.all_finite())
        {
            done(program, state, epsilon, logger);
            break;
        }

        // backtracking line-search: stage 1
        auto s    = s0 * make_smax(state.m_u, du);
        auto iter = tensor_size_t{0};
        for (iter = 0; iter < max_lsearch_iters; ++iter)
        {
            if ((G * (state.m_x + s * dx) - h).maxCoeff() < 0.0)
            {
                break;
            }
            else
            {
                s *= beta;
            }
        }
        if (iter == max_lsearch_iters)
        {
            done(program, state, epsilon, logger);
            break;
        }

        // backtracking line-search: stage 2
        const auto r0 = state.residual();
        for (iter = 0; iter < max_lsearch_iters; ++iter)
        {
            program.update(state.m_x + s * dx, state.m_u + s * du, state.m_v + s * dv, miu, state);
            if (state.residual() <= (1.0 - alpha * s) * r0)
            {
                break;
            }
            else
            {
                s *= beta;
            }
        }
        if (iter == max_lsearch_iters)
        {
            // NB: revert to previous state if the residual didn't improve!
            if (state.residual() > r0)
            {
                program.update(state.m_x, state.m_u, state.m_v, miu, state);
            }
            done(program, state, epsilon, logger);
            break;
        }

        // update state
        state.m_x += s * dx;
        state.m_u += s * du;
        state.m_v += s * dv;
        state.update(program.m_Q, program.m_c, program.m_A, program.m_b, program.m_G, program.m_h);

        const auto curr_eta   = state.m_eta;
        const auto curr_rdual = state.m_rdual.lpNorm<2>();
        const auto curr_rprim = state.m_rprim.lpNorm<2>();

        // check stopping criteria
        if (!std::isfinite(curr_eta) || !std::isfinite(curr_rdual) || !std::isfinite(curr_rprim))
        {
            // numerical instabilities
            state.m_status = solver_status::failed;
            logger.info("[program]: ", state, ",feasible=", program.feasible(state), ".\n");
            break;
        }
        else if (std::max({prev_eta - curr_eta, prev_rdual - curr_rdual, prev_rprim - curr_rprim}) < epsilon0)
        {
            // very precise convergence detected, check global convergence criterion!
            done(program, state, epsilon, logger);
            break;
        }
        else
        {
            logger.info("[program]: ", state, ",feasible=", program.feasible(state), ".\n");
        }
    }

    return state;
}

solver_state_t solver_t::solve_without_inequality(const program_t& program, const logger_t& logger) const
{
    const auto miu = parameter("solver::miu").value<scalar_t>();

    const auto& c = program.m_c;
    const auto& b = program.m_b;
    const auto  n = program.n();
    const auto  p = program.p();

    // NB: solve directly the KKT-based system of linear equations coupling (x, v)
    program.solve(matrix_t::zero(n, n), c, -b);

    // current solver state
    auto state  = solver_state_t{n, 0, p};
    state.m_x   = program.m_lsol.segment(0, n);
    state.m_v   = program.m_lsol.segment(n, p);
    state.m_eta = 0.0;
    program.update(state.m_x, state.m_u, state.m_v, miu, state);
    state.update(program.m_Q, program.m_c, program.m_A, program.m_b, program.m_G, program.m_h);

    const auto valid = std::isfinite(state.residual());
    const auto aprox = (program.m_lmat * program.m_lsol).isApprox(program.m_lvec.vector(), epsilon2<scalar_t>());
    state.m_status =
        (valid && aprox) ? solver_status::converged : (!valid ? solver_status::failed : solver_status::unfeasible);

    logger.info("[program]: ", state, ".\n");
    return state;
}

void solver_t::done(const program_t& program, solver_state_t& state, const scalar_t epsilon, const logger_t& logger)
{
    const auto feasible = program.feasible(state);

    if (feasible && std::max({state.m_eta, state.m_rdual.lpNorm<2>(), state.m_rprim.lpNorm<2>()}) < epsilon)
    {
        state.m_status = solver_status::converged;
    }
    else
    {
        // FIXME: this is an heuristic, to search for a theoretically sound method to detect unboundness and
        // unfeasibility!
        state.m_status = feasible ? solver_status::unbounded : solver_status::unfeasible;
    }

    logger.info("[program]: ", state, ",feasible=", feasible, ".\n");
}
