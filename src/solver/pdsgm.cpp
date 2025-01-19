#include <solver/pdsgm.h>

using namespace nano;

struct solver_pdsgm_t::model_t
{
    explicit model_t(const vector_t& x0, const scalar_t D)
        : m_x0(x0)
        , m_D(D)
        , m_sk1(x0.size())
        , m_xk1h(x0.size())
    {
        m_sk1.array()  = 0.0;
        m_xk1h.array() = 0.0;
    }

    void reset(const scalar_t D)
    {
        m_D = D;
        m_L = 0.0;
    }

    void updateL(const vector_t& gxk)
    {
        // NB: update the estimation of the Lipschitz constant.
        // NB: reset state when a gradient with a larger magnitude is found.
        const auto gnorm = gxk.lpNorm<2>();
        if (gnorm > m_L)
        {
            m_L            = gnorm;
            m_sk1.array()  = 0.0;
            m_xk1h.array() = 0.0;
            m_lgx          = 0.0;
            m_Sk           = 0.0;
            m_beta         = 1.0;
        }
    }

    void update(const scalar_t lambdak, const vector_t& xk, const vector_t& gxk)
    {
        m_beta = m_beta + 1.0 / m_beta;
        m_Sk += lambdak;
        m_xk1h += lambdak * xk;
        m_sk1 += lambdak * gxk;
        m_lgx += lambdak * gxk.dot(xk - m_x0);
    }

    auto gap() const { return (m_lgx + std::sqrt(2.0 * m_D * m_sk1.dot(m_sk1))) / m_Sk; }

    auto xk1(const scalar_t beta) const { return m_x0 - m_sk1 / beta; }

    auto dual_xk1() const { return m_xk1h / m_Sk; }

    const vector_t& m_x0;
    scalar_t        m_D{1.0};
    scalar_t        m_L{0.0};
    vector_t        m_sk1;
    vector_t        m_xk1h;
    scalar_t        m_Sk{0.0};
    scalar_t        m_lgx{0.0};
    scalar_t        m_beta{1.0};
};

solver_pdsgm_t::solver_pdsgm_t(string_t id)
    : solver_t(std::move(id))
{
    static constexpr auto fmax = std::numeric_limits<scalar_t>::max();

    register_parameter(parameter_t::make_scalar("solver::pdsgm::D", 0.0, LT, 1e+0, LE, fmax));
}

solver_state_t solver_pdsgm_t::do_minimize(const function_t& function, const vector_t& x0, const logger_t& logger) const
{
    warn_nonsmooth(function, logger);
    warn_constrained(function, logger);

    const auto max_evals = parameter("solver::max_evals").value<tensor_size_t>();
    const auto D         = parameter("solver::pdsgm::D").value<scalar_t>();

    auto state = solver_state_t{function, x0}; // NB: keeps track of the best state

    auto x     = state.x();
    auto gx    = state.gx();
    auto model = model_t{x0, D};

    while (function.fcalls() + function.gcalls() < max_evals)
    {
        if (gx.lpNorm<Eigen::Infinity>() < std::numeric_limits<scalar_t>::epsilon())
        {
            const auto iter_ok = state.valid();
            solver_t::done_gradient_test(state, iter_ok, logger);
            break;
        }

        model.updateL(gx);
        const auto [lambda, betah] = update(model, gx);
        model.update(lambda, x, gx);

        x             = model.xk1(betah);
        const auto fx = function(x, gx);
        state.update_if_better(x, gx, fx); // FIXME: option to obtain only the gradient without the function value!

        const auto iter_ok = std::isfinite(fx);
        if (solver_t::done_value_test(state, iter_ok, logger))
        {
            break;
        }
    }

    return state;
}

solver_sda_t::solver_sda_t()
    : solver_pdsgm_t("sda")
{
}

rsolver_t solver_sda_t::clone() const
{
    return std::make_unique<solver_sda_t>(*this);
}

std::tuple<scalar_t, scalar_t> solver_sda_t::update(const model_t& model, const vector_t&) const
{
    const auto gamma = model.m_L / std::sqrt(2.0 * model.m_D);

    return std::make_tuple(1.0, gamma * model.m_beta);
}

solver_wda_t::solver_wda_t()
    : solver_pdsgm_t("wda")
{
}

rsolver_t solver_wda_t::clone() const
{
    return std::make_unique<solver_wda_t>(*this);
}

std::tuple<scalar_t, scalar_t> solver_wda_t::update(const model_t& model, const vector_t& gx) const
{
    const auto ro = std::sqrt(2.0 * model.m_D);

    return std::make_tuple(1.0 / gx.lpNorm<2>(), model.m_beta / ro);
}
