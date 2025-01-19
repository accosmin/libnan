#include <nano/critical.h>
#include <nano/solver.h>
#include <nano/tuner/surrogate.h>
#include <nano/tuner/util.h>

using namespace nano;

quadratic_surrogate_fit_t::quadratic_surrogate_fit_t(const loss_t& loss, tensor2d_t p, tensor1d_t y)
    : function_t("quadratic surrogate fitting function", (p.cols() + 1) * (p.cols() + 2) / 2)
    , m_loss(loss)
    , m_p2(p.size<0>(), size())
    , m_y(std::move(y))
    , m_loss_outputs(p.size<0>(), 1, 1, 1)
    , m_loss_values(p.size<0>())
    , m_loss_vgrads(p.size<0>(), 1, 1, 1)
{
    convex(loss.convex() ? convexity::yes : convexity::no);
    smooth(loss.smooth() ? smoothness::yes : smoothness::no);

    assert(m_p2.size<0>() == m_y.size<0>());

    for (tensor_size_t sample = 0, samples = p.size<0>(), size = p.size<1>(); sample < samples; ++sample)
    {
        auto k = tensor_size_t{0};

        m_p2(sample, k++) = 1.0;
        for (tensor_size_t i = 0; i < size; ++i)
        {
            m_p2(sample, k++) = p(sample, i);
        }
        for (tensor_size_t i = 0; i < size; ++i)
        {
            for (tensor_size_t j = i; j < size; ++j)
            {
                m_p2(sample, k++) = p(sample, i) * p(sample, j);
            }
        }
    }
}

rfunction_t quadratic_surrogate_fit_t::clone() const
{
    return std::make_unique<quadratic_surrogate_fit_t>(*this);
}

scalar_t quadratic_surrogate_fit_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    m_loss_outputs.vector() = m_p2.matrix() * x;

    const auto samples = m_p2.size<0>();
    const auto targets = m_y.reshape(samples, 1, 1, 1);

    if (gx.size() == x.size())
    {
        m_loss.vgrad(targets, m_loss_outputs, m_loss_vgrads);
        gx = m_p2.matrix().transpose() * m_loss_vgrads.vector();
    }

    m_loss.value(targets, m_loss_outputs, m_loss_values);
    return m_loss_values.sum();
}

quadratic_surrogate_t::quadratic_surrogate_t(vector_t model)
    : function_t("quadratic surrogate function", static_cast<tensor_size_t>(std::sqrt(2 * model.size())) - 1)
    , m_model(std::move(model))
{
    convex(convexity::no);
    smooth(smoothness::yes);

    assert(size() > 0);
    assert(m_model.size() == (size() + 1) * (size() + 2) / 2);
}

rfunction_t quadratic_surrogate_t::clone() const
{
    return std::make_unique<quadratic_surrogate_t>(*this);
}

scalar_t quadratic_surrogate_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    if (gx.size() == x.size())
    {
        gx.zero();

        auto k = tensor_size_t{1};
        for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
        {
            gx(i) += m_model(k++);
        }
        for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
        {
            for (tensor_size_t j = i; j < size; ++j)
            {
                gx(i) += m_model(k) * x(j);
                gx(j) += m_model(k++) * x(i);
            }
        }
    }

    scalar_t fx = m_model(0);

    auto k = tensor_size_t{1};
    for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
    {
        fx += m_model(k++) * x(i);
    }
    for (tensor_size_t i = 0, size = x.size(); i < size; ++i)
    {
        for (tensor_size_t j = i; j < size; ++j)
        {
            fx += m_model(k++) * x(i) * x(j);
        }
    }
    return fx;
}

surrogate_tuner_t::surrogate_tuner_t()
    : tuner_t("surrogate")
{
}

rtuner_t surrogate_tuner_t::clone() const
{
    return std::make_unique<surrogate_tuner_t>(*this);
}

void surrogate_tuner_t::do_optimize(const param_spaces_t& spaces, const tuner_callback_t& callback,
                                    const logger_t& logger, tuner_steps_t& steps) const
{
    const auto max_evals = parameter("tuner::max_evals").value<size_t>();
    const auto min_igrid = make_min_igrid(spaces);
    const auto max_igrid = make_max_igrid(spaces);

    // fit and optimize the surrogate model iteratively...
    const auto loss = loss_t::all().get("mse");
    assert(loss != nullptr);

    const auto solver = solver_t::all().get("lbfgs");
    assert(solver != nullptr);

    const auto to_surrogate = [&](const auto& values)
    {
        tensor1d_t result(values.size());
        for (tensor_size_t i = 0, size = values.size(); i < size; ++i)
        {
            result(i) = spaces[static_cast<size_t>(i)].to_surrogate(values(i));
        }
        return result;
    };

    for (; !steps.empty() && steps.size() < max_evals;)
    {
        tensor2d_t p(static_cast<tensor_size_t>(steps.size()), static_cast<tensor_size_t>(spaces.size()));
        tensor1d_t y(static_cast<tensor_size_t>(steps.size()));

        tensor_size_t k = 0;
        for (const auto& step : steps)
        {
            p.tensor(k) = to_surrogate(step.m_param);
            y(k++)      = step.m_value;
        }

        const auto surrogate_fit = quadratic_surrogate_fit_t{*loss, p, y};
        const auto min_state_fit = solver->minimize(surrogate_fit, vector_t::zero(surrogate_fit.size()), logger);
        critical(min_state_fit.valid(), "tuner: failed to fit the surrogate model <", min_state_fit, ">!");

        const auto surrogate_opt = quadratic_surrogate_t{min_state_fit.x()};
        const auto min_state_opt =
            solver->minimize(surrogate_opt, to_surrogate(steps.begin()->m_param).vector(), logger);
        critical(min_state_opt.valid(), "tuner: failed to optimize the surrogate model <", min_state_opt, ">!");

        const auto& min_state_opt_x = min_state_opt.x();

        indices_t src_igrid(min_state_opt_x.size());
        for (tensor_size_t iparam = 0; iparam < min_state_opt_x.size(); ++iparam)
        {
            const auto& space = spaces[static_cast<size_t>(iparam)];
            src_igrid(iparam) = space.closest_grid_point_from_surrogate(min_state_opt_x(iparam));
        }

        const auto igrids = local_search(min_igrid, max_igrid, src_igrid, 1);
        if (!evaluate(spaces, callback, igrids, logger, steps))
        {
            break;
        }
    }
}
