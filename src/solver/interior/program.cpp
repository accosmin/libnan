#include <nano/function/util.h>
#include <solver/interior/program.h>

using namespace nano;

program_t::program_t(const linear_program_t& program, linear_constraints_t constraints)
    : program_t(program, matrix_t{}, program.c(), std::move(constraints))
{
}

program_t::program_t(const quadratic_program_t& program, linear_constraints_t constraints)
    : program_t(program, program.Q(), program.c(), std::move(constraints))
{
}

program_t::program_t(const function_t& function, matrix_t Q, vector_t c, linear_constraints_t constraints)
    : m_function(function)
    , m_Q(std::move(Q))
    , m_c(std::move(c))
    , m_A(std::move(constraints.m_A))
    , m_b(std::move(constraints.m_b))
    , m_G(std::move(constraints.m_G))
    , m_h(std::move(constraints.m_h))
    , m_lmat(n() + p(), n() + p())
    , m_lvec(n() + p())
    , m_lsol(n() + p())
{
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

const vector_t& program_t::solve() const
{
    m_ldlt.compute(m_lmat.matrix());
    m_lsol.vector() = m_ldlt.solve(m_lvec.vector());
    return m_lsol;
}

void program_t::update(const scalar_t s, const scalar_t miu, state_t& state) const
{
    const auto m = this->m();
    const auto p = this->p();
    const auto x = state.m_x + s * state.m_dx;
    const auto u = state.m_u + s * state.m_du;
    const auto v = state.m_v + s * state.m_dv;

    // objective
    if (!m_Q.size())
    {
        state.m_rdual = m_c;
    }
    else
    {
        state.m_rdual = Q() * x + m_c;
    }

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

bool program_t::valid(const scalar_t epsilon) const
{
    return m_lmat.all_finite() && m_lvec.all_finite() && m_lsol.all_finite() &&
           (m_lmat * m_lsol - m_lvec).lpNorm<Eigen::Infinity>() < epsilon;
}
