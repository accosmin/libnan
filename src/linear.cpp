#include <nano/linear/elastic_net.h>
#include <nano/linear/function.h>
#include <nano/linear/lasso.h>
#include <nano/linear/ordinary.h>
#include <nano/linear/result.h>
#include <nano/linear/ridge.h>
#include <nano/linear/util.h>
#include <nano/machine/tune.h>
#include <nano/tensor/stream.h>

using namespace nano;
using namespace nano::linear;

namespace
{
auto make_x0(const ::nano::linear::function_t& function, const std::any& extra)
{
    vector_t x0 = vector_t::zero(function.size());
    if (extra.has_value())
    {
        const auto& result                      = std::any_cast<linear::result_t>(extra);
        const auto& bias                        = result.m_bias;
        const auto& weights                     = result.m_weights;
        x0.segment(0, weights.size())           = weights.array();
        x0.segment(weights.size(), bias.size()) = bias.array();
    }
    return x0;
} // LCOV_EXCL_LINE

auto fit(const linear_t& model, const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
         const solver_t& solver, tensor1d_cmap_t params, const logger_t& logger, const std::any& extra = std::any{})
{
    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(model.parameter("linear::batch").value<tensor_size_t>());
    iterator.scaling(model.parameter("linear::scaling").value<scaling_type>());
    iterator.cache_flatten(std::numeric_limits<tensor_size_t>::max());
    iterator.cache_targets(std::numeric_limits<tensor_size_t>::max());

    const auto function = model.make_function(iterator, loss, params);
    const auto state    = solver.minimize(function, make_x0(function, extra), logger);

    tensor1d_t bias    = function.bias(state.x());
    tensor2d_t weights = function.weights(state.x());
    ::upscale(iterator.flatten_stats(), iterator.scaling(), iterator.targets_stats(), iterator.scaling(), weights,
              bias);

    return linear::result_t{std::move(bias), std::move(weights), state};
}
} // namespace

linear_t::linear_t(string_t id)
    : typed_t(std::move(id))
{
    register_parameter(parameter_t::make_integer("linear::batch", 10, LE, 100, LE, 10000));
    register_parameter(parameter_t::make_enum("linear::scaling", scaling_type::standard));
}

std::istream& linear_t::read(std::istream& stream)
{
    learner_t::read(stream);

    critical(::nano::read(stream, m_bias) && ::nano::read(stream, m_weights), "linear: failed to read from stream!");

    critical(m_bias.size() == m_weights.rows(), "linear: parameter mismatch!");

    return stream;
}

std::ostream& linear_t::write(std::ostream& stream) const
{
    learner_t::write(stream);

    critical(::nano::write(stream, m_bias) && ::nano::write(stream, m_weights), "linear: failed to write to stream!");

    return stream;
}

ml::result_t linear_t::fit(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                           const ml::params_t& fit_params)
{
    learner_t::fit_dataset(dataset);

    const auto batch = parameter("linear::batch").value<tensor_size_t>();

    // tune hyper-parameters (if any)
    const auto callback = [&](const indices_t& train_samples, const indices_t& valid_samples,
                              const tensor1d_cmap_t params, const std::any& extra, const logger_t& logger)
    {
        auto result    = ::fit(*this, dataset, train_samples, loss, fit_params.solver(), params, logger, extra);
        auto tr_values = ::nano::linear::evaluate(dataset, train_samples, loss, result.m_weights, result.m_bias, batch);
        auto vd_values = ::nano::linear::evaluate(dataset, valid_samples, loss, result.m_weights, result.m_bias, batch);

        return std::make_tuple(std::move(tr_values), std::move(vd_values), std::move(result));
    };
    auto fit_result = ml::tune("linear", samples, fit_params, make_param_spaces(), callback);

    // refit with the optimum hyper-parameters (if any) on all given samples
    {
        const auto logger = make_file_logger(fit_result.refit_log_path());
        const auto params = fit_result.params(fit_result.optimum_trial());

        auto result = ::fit(*this, dataset, samples, loss, fit_params.solver(), params, logger);
        auto values = ::nano::linear::evaluate(dataset, samples, loss, result.m_weights, result.m_bias, batch);

        m_bias    = result.m_bias;
        m_weights = result.m_weights;

        fit_result.store(std::move(values), std::move(result));
    }
    fit_params.log(fit_result, fit_result.trials(), "linear");

    return fit_result;
}

void linear_t::do_predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const
{
    // TODO: no need to allocate the sample indices one more time
    auto iterator = flatten_iterator_t(dataset, samples);
    iterator.scaling(scaling_type::none);
    iterator.batch(parameter("linear::batch").value<tensor_size_t>());

    iterator.loop([&](tensor_range_t range, size_t, tensor2d_cmap_t inputs)
                  { ::nano::linear::predict(inputs, m_weights, m_bias, outputs.slice(range)); });
}

factory_t<linear_t>& linear_t::all()
{
    static auto manager = factory_t<linear_t>{};
    const auto  op      = []()
    {
        manager.add<ordinary_t>("linear model (with no regularization)");
        manager.add<lasso_t>("linear model regularized with the L1-norm of the weights");
        manager.add<ridge_t>("linear model regularized with the L2-norm of the weights");
        manager.add<elastic_net_t>("linear model regularized with both the L1 and the L2-norm of the weights");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
