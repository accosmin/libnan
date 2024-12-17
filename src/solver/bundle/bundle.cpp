#include <solver/bundle/bundle.h>

using namespace nano;

namespace
{
auto make_program(const tensor_size_t n)
{
    auto Q = matrix_t{matrix_t::zero(n + 1, n + 1)};
    auto c = vector_t{vector_t::zero(n + 1)};
    return program::quadratic_program_t{std::move(Q), std::move(c)};
}

template <typename tvectory>
auto eval_cutting_planes(matrix_cmap_t G, vector_cmap_t h, const tvectory& y)
{
    const auto m = h.size();
    const auto n = y.size();
    assert(G.rows() == m);
    assert(G.cols() == n + 1);

    auto value = std::numeric_limits<scalar_t>::lowest();
    for (tensor_size_t i = 0; i < m; ++i)
    {
        value = std::max(value, h(i) + G.vector(i).segment(0, n).dot(y));
    }
    return value;
}

template <typename tvectory>
void write_cutting_plane(vector_map_t g, scalar_t& h, const vector_cmap_t x, const vector_cmap_t y, const tvectory& gy,
                         const scalar_t fy)
{
    const auto n = x.size();

    assert(y.size() == n);
    assert(gy.size() == n);
    assert(g.size() == n + 1);

    h               = fy + gy.dot(x - y);
    g.segment(0, n) = gy.array();
    g(n)            = -1.0;
}
} // namespace

bundle_t::solution_t::solution_t(const tensor_size_t dims)
    : m_x(dims)
{
}

bundle_t::bundle_t(const solver_state_t& state, const tensor_size_t max_size)
    : m_program(make_program(state.x().size()))
    , m_bundleG(max_size + 1, state.x().size() + 1)
    , m_bundleH(max_size + 1)
    , m_solution(state.x().size())
    , m_x(state.x())
    , m_gx(state.gx())
    , m_fx(state.fx())
{
    append(state.x(), state.gx(), state.fx(), true);
}

scalar_t bundle_t::etol(const scalar_t epsilon) const
{
    return epsilon * std::sqrt(static_cast<size_t>(dims()));
}

scalar_t bundle_t::gtol(const scalar_t epsilon) const
{
    return epsilon * std::sqrt(static_cast<size_t>(dims()));
}

void bundle_t::moveto(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy)
{
    const auto serious_step = true;
    append(y, gy, fy, serious_step);
}

void bundle_t::append(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy)
{
    const auto serious_step = false;
    append(y, gy, fy, serious_step);
}

const bundle_t::solution_t& bundle_t::solve(const scalar_t tau, const scalar_t level, const logger_t& logger)
{
    assert(size() > 0);
    assert(dims() == m_x.size());

    const auto n         = dims();
    const auto m         = size();
    const auto has_level = std::isfinite(level);

    const auto bundleG = m_bundleG.slice(0, m);
    const auto bundleH = m_bundleH.slice(0, m);
    const auto bundleF = m_fx - bundleH.array();

    logger.info("bundle: H=", bundleH.array(), ",tau=", tau, ".\n");

    // construct quadratic programming problem
    // NB: equivalent and simpler problem is to solve for `y = x - x_k^`!
    m_program.m_Q.block(0, 0, n, n).diagonal().array() = 1.0 / tau;
    m_program.m_c(n)                                   = 1.0;

    if (has_level)
    {
        auto weights    = m_bundleG.vector(capacity() - 1);
        weights.array() = 0.0;
        weights(n)      = 1.0;
        m_program.constrain(program::make_inequality(bundleG, bundleF), program::make_inequality(weights, level));
    }
    else
    {
        m_program.constrain(program::make_inequality(bundleG, bundleF));
    }

    // solve for (y, r) => (x = y + x_k^, r)!
    // FIXME: can estimate a reasonable starting point?!
    const auto solution = m_solver.solve(m_program, logger);
    assert(solution.m_x.size() == n + 1);

    if (!m_program.feasible(solution.m_x, epsilon0<scalar_t>()))
    {
        logger.error("bundle: unfeasible solution, deviation(ineq)=", m_program.m_ineq.deviation(solution.m_x), ".\n");
    }

    // NB: the quadratic program may be unfeasible, so the level needs to be moved towards the stability center!
    if (solution.m_status != solver_status::converged && !has_level)
    {
        logger.error("bundle: failed to solve, status=", solution.m_status, ".\n");
    }

    // extract solution and statistics, see (1)
    assert(solution.m_u.size() == (has_level ? (m + 1) : m));

    m_solution.m_x      = solution.m_x.segment(0, n) + m_x;
    m_solution.m_r      = solution.m_x(n) + m_fx;
    m_solution.m_tau    = tau;
    m_solution.m_alphas = solution.m_u.segment(0, m);
    m_solution.m_lambda = has_level ? solution.m_u(m) : 0.0;

    assert(m_solution.m_alphas.min() >= 0.0);

    return m_solution;
}

scalar_t bundle_t::fhat(const vector_t& x) const
{
    assert(size() > 0);
    assert(dims() == x.size());

    const auto m       = size();
    const auto bundleG = m_bundleG.slice(0, m);
    const auto bundleH = m_bundleH.slice(0, m);

    return eval_cutting_planes(bundleG, bundleH, x - m_x);
}

void bundle_t::delete_inactive(const scalar_t epsilon)
{
    if (size() > 0)
    {
        m_bsize = remove_if([&](const tensor_size_t i) { return m_solution.m_alphas(i) < epsilon; });
    }
}

void bundle_t::delete_smallest(const tensor_size_t count)
{
    if (size() + 1 == capacity())
    {
        store_aggregate();

        // see (1), ch 5.1.4 - remove the linearizations with the smallest Lagrange multipliers!
        // NB: reuse the alphas buffer as it will be re-computed anyway at the next proximal point update!
        [[maybe_unused]] const auto old_size = m_bsize;
        assert(count <= m_bsize);

        std::vector<std::pair<scalar_t, tensor_size_t>> alphas;
        alphas.reserve(static_cast<size_t>(m_bsize));
        for (tensor_size_t i = 0; i < m_bsize; ++i)
        {
            alphas.emplace_back(m_solution.m_alphas(i), i);
        }

        std::sort(alphas.begin(), alphas.end());
        alphas.erase(alphas.begin() + count, alphas.end());

        assert(alphas.size() == static_cast<size_t>(count));
        [[maybe_unused]] const auto threshold = alphas.rbegin()->first;

        m_bsize = remove_if(
            [&](const tensor_size_t i)
            {
                const auto op = [&](const auto& ialpha) { return ialpha.second == i; };
                const auto it = std::find_if(alphas.begin(), alphas.end(), op);
                return it != alphas.end();
            });

        assert(m_bsize + count == old_size);
        assert(m_solution.m_alphas.slice(0, m_bsize).min() >= threshold);

        append_aggregate();
    }
}

void bundle_t::store_aggregate()
{
    // NB: stored the aggregation in the last slot!
    const auto ilast = capacity() - 1;
    const auto fhat  = this->fhat(m_solution.m_x);
    const auto ghat  = (m_x - m_solution.m_x) / m_solution.m_tau;
    write_cutting_plane(m_bundleG.tensor(ilast), m_bundleH(ilast), m_x, m_solution.m_x, ghat, fhat);
}

void bundle_t::append_aggregate()
{
    // NB: load the aggregation from the last slot!
    const auto ilast          = capacity() - 1;
    m_bundleH(m_bsize)        = m_bundleH(ilast);
    m_bundleG.vector(m_bsize) = m_bundleG.vector(ilast);
    ++m_bsize;
}

void bundle_t::append(const vector_cmap_t y, const vector_cmap_t gy, const scalar_t fy, const bool serious_step)
{
    assert(dims() == y.size());
    assert(dims() == gy.size());

    delete_inactive(epsilon0<scalar_t>());
    delete_smallest(2);

    for (tensor_size_t i = 0; serious_step && i < m_bsize; ++i)
    {
        m_bundleH(i) += m_bundleG.row(i).segment(0, dims()).dot(y - m_x);
    }

    if (serious_step)
    {
        m_x  = y;
        m_gx = gy;
        m_fx = fy;
    }

    write_cutting_plane(m_bundleG.tensor(m_bsize), m_bundleH(m_bsize), m_x, y, gy, fy);
    ++m_bsize;

    assert(m_bsize < capacity());
}

void bundle_t::config(configurable_t& c, const string_t& prefix)
{
    c.register_parameter(parameter_t::make_integer(scat(prefix, "::bundle::max_size"), 2, LE, 100, LE, 1000));
}

bundle_t bundle_t::make(const solver_state_t& state, const configurable_t& c, const string_t& prefix)
{
    const auto max_size = c.parameter(scat(prefix, "::bundle::max_size")).value<tensor_size_t>();

    return {state, max_size};
}
