#include <nano/gboost/early_stopping.h>
#include <nano/gboost/enums.h>
#include <nano/gboost/function.h>
#include <nano/gboost/model.h>
#include <nano/gboost/result.h>
#include <nano/gboost/sampler.h>
#include <nano/gboost/util.h>
#include <nano/machine/tune.h>
#include <nano/tensor/stream.h>
#include <nano/wlearner/util.h>
#include <set>

using namespace nano;
using namespace nano::gboost;

namespace
{
auto make_params(const configurable_t& configurable)
{
    auto param_spaces = param_spaces_t{};

    if (const auto shrinkage = configurable.parameter("gboost::shrinkage").value<gboost_shrinkage>();
        shrinkage == gboost_shrinkage::global)
    {
        param_spaces.emplace_back("shrinkage", param_space_t::type::linear, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                  1.0);
    }

    return param_spaces;
}

auto decode_params(const tensor1d_cmap_t& params, const gboost_shrinkage shrinkage)
{
    scalar_t shrinkage_ratio = 1.0;

    tensor_size_t index = 0;
    if (shrinkage == gboost_shrinkage::global)
    {
        shrinkage_ratio = params(index);
        index++;
    }

    return std::make_tuple(shrinkage_ratio);
}

auto selected(const tensor2d_t& values, const indices_t& samples)
{
    auto selected = tensor2d_t{2, samples.size()};
    values.tensor(0).indexed(samples, selected.tensor(0));
    values.tensor(1).indexed(samples, selected.tensor(1));
    return selected;
}

auto make_cluster(const dataset_t& dataset, const indices_t& samples, const wlearner_t& wlearner,
                  const gboost_wscale wscale)
{
    if (wscale == gboost_wscale::tboost)
    {
        return wlearner.split(dataset, samples);
    }
    else
    {
        cluster_t cluster{dataset.samples(), 1};
        for (tensor_size_t i = 0; i < samples.size(); ++i)
        {
            cluster.assign(samples(i), 0);
        }
        return cluster;
    }
}

auto fit(const configurable_t& configurable, const dataset_t& dataset, const indices_t& train_samples,
         const indices_t& valid_samples, const loss_t& loss, const solver_t& solver, const rwlearners_t& prototypes,
         const tensor1d_t& params, const logger_t& logger)
{
    const auto seed            = configurable.parameter("gboost::seed").value<uint64_t>();
    const auto batch           = configurable.parameter("gboost::batch").value<tensor_size_t>();
    const auto epsilon         = configurable.parameter("gboost::epsilon").value<scalar_t>();
    const auto patience        = configurable.parameter("gboost::patience").value<size_t>();
    auto       max_rounds      = configurable.parameter("gboost::max_rounds").value<tensor_size_t>();
    const auto wscale          = configurable.parameter("gboost::wscale").value<gboost_wscale>();
    const auto subsample       = configurable.parameter("gboost::subsample").value<gboost_subsample>();
    const auto shrinkage       = configurable.parameter("gboost::shrinkage").value<gboost_shrinkage>();
    const auto subsample_ratio = configurable.parameter("gboost::subsample_ratio").value<scalar_t>();

    auto [shrinkage_ratio] = decode_params(params, shrinkage);

    const auto samples = arange(0, dataset.samples());

    auto targets_iterator = targets_iterator_t{dataset, samples};
    targets_iterator.batch(batch);
    targets_iterator.scaling(scaling_type::none);

    auto train_targets_iterator = targets_iterator_t{dataset, train_samples};
    train_targets_iterator.batch(batch);
    train_targets_iterator.scaling(scaling_type::none);

    auto valid_targets_iterator = targets_iterator_t{dataset, valid_samples};
    valid_targets_iterator.batch(batch);
    valid_targets_iterator.scaling(scaling_type::none);

    auto sampler  = sampler_t{train_samples, subsample, seed, subsample_ratio};
    auto values   = tensor2d_t{2, samples.size()};
    auto outputs  = tensor4d_t{cat_dims(samples.size(), dataset.target_dims())};
    auto woutputs = tensor4d_t{cat_dims(samples.size(), dataset.target_dims())};

    const auto gfunction = grads_function_t{targets_iterator, loss};
    const auto bfunction = bias_function_t{train_targets_iterator, loss};

    // estimate bias
    const auto bstate = solver.minimize(bfunction, make_full_vector<scalar_t>(bfunction.size(), 0.0), logger);

    outputs.reshape(samples.size(), -1).matrix().rowwise() = bstate.x().transpose();
    ::nano::gboost::evaluate(targets_iterator, loss, outputs, values);

    auto result   = gboost::result_t{&values, &train_samples, &valid_samples, max_rounds + 1};
    result.m_bias = map_tensor(bstate.x().data(), make_dims(bstate.x().size()));
    result.update(0, shrinkage_ratio, bstate);

    // keep track of the optimum boosting round using the validation error
    auto optimum = early_stopping_t{values};
    if (optimum.done(values, train_samples, valid_samples, result.m_wlearners, epsilon, patience))
    {
        max_rounds = 0;
    }

    // construct the model one boosting round at a time
    for (tensor_size_t round = 0; round < max_rounds; ++round)
    {
        const auto& gradients   = gfunction.gradients(outputs);
        const auto  fit_samples = sampler.sample(values, gradients);

        // choose the weak learner that aligns the best with the current residuals
        auto best_score    = wlearner_t::no_fit_score();
        auto best_wlearner = rwlearner_t{};
        for (const auto& prototype : prototypes)
        {
            auto       wlearner = prototype->clone();
            const auto score    = wlearner->fit(dataset, fit_samples, gradients);
            if (score < best_score)
            {
                best_score    = score;
                best_wlearner = std::move(wlearner);
            }
        }
        if (!best_wlearner)
        {
            break;
        }

        // scale the chosen weak learner
        woutputs.zero();
        best_wlearner->predict(dataset, samples, woutputs.tensor());

        const auto cluster  = make_cluster(dataset, samples, *best_wlearner, wscale);
        const auto function = scale_function_t{train_targets_iterator, loss, cluster, outputs, woutputs};

        auto gstate = solver.minimize(function, make_full_vector<scalar_t>(function.size(), 1.0), logger);
        if (gstate.x().min() < std::numeric_limits<scalar_t>::epsilon())
        {
            // NB: scaling fails (optimization fails or convergence on training loss)
            result.update(round + 1, shrinkage_ratio, gstate, std::move(best_wlearner));
            break;
        }

        // apply shrinkage
        best_wlearner->scale(gstate.x() * shrinkage_ratio);
        woutputs.zero();
        best_wlearner->predict(dataset, samples, woutputs.tensor());

        if (shrinkage == gboost_shrinkage::local)
        {
            shrinkage_ratio = tune_shrinkage(valid_targets_iterator, loss, outputs, woutputs);
            best_wlearner->scale(make_full_vector<scalar_t>(1, shrinkage_ratio));
            woutputs.array() *= shrinkage_ratio;
        }

        // update predictions
        outputs.vector() += woutputs.vector();
        ::nano::gboost::evaluate(targets_iterator, loss, outputs, values);
        result.update(round + 1, shrinkage_ratio, gstate, std::move(best_wlearner));

        // early stopping
        if (optimum.done(values, train_samples, valid_samples, result.m_wlearners, epsilon, patience))
        {
            break;
        }
    }

    result.done(static_cast<tensor_size_t>(optimum.round()));

    return std::make_tuple(std::move(result), selected(optimum.values(), train_samples),
                           selected(optimum.values(), valid_samples));
}
} // namespace

gboost_model_t::gboost_model_t()
{
    register_parameter(parameter_t::make_scalar("gboost::epsilon", 1e-12, LE, 1e-6, LE, 1.0));

    register_parameter(parameter_t::make_integer("gboost::seed", 0, LE, 42, LE, 1024));
    register_parameter(parameter_t::make_integer("gboost::batch", 10, LE, 100, LE, 10000));
    register_parameter(parameter_t::make_integer("gboost::patience", 1, LE, 10, LE, 1'000));
    register_parameter(parameter_t::make_integer("gboost::max_rounds", 10, LE, 1'000, LE, 1'000'000));

    register_parameter(parameter_t::make_enum("gboost::wscale", gboost_wscale::gboost));
    register_parameter(parameter_t::make_enum("gboost::shrinkage", gboost_shrinkage::off));
    register_parameter(parameter_t::make_enum("gboost::subsample", gboost_subsample::off));
    register_parameter(parameter_t::make_scalar("gboost::subsample_ratio", 0.0, LT, 1.0, LE, 1.0));
}

gboost_model_t::gboost_model_t(gboost_model_t&&) noexcept = default;

gboost_model_t& gboost_model_t::operator=(gboost_model_t&&) noexcept = default;

gboost_model_t::gboost_model_t(const gboost_model_t& other)
    : learner_t(other)
    , m_bias(other.m_bias)
    , m_wlearners(wlearner::clone(other.m_wlearners))
    , m_prototypes(wlearner::clone(other.m_prototypes))
{
}

gboost_model_t& gboost_model_t::operator=(const gboost_model_t& other)
{
    if (this != &other)
    {
        learner_t::operator=(other);
        m_bias       = other.m_bias;
        m_wlearners  = wlearner::clone(other.m_wlearners);
        m_prototypes = wlearner::clone(other.m_prototypes);
    }
    return *this;
}

gboost_model_t::~gboost_model_t() = default;

void gboost_model_t::prototypes(const rwlearners_t& prototypes)
{
    m_prototypes = wlearner::clone(prototypes);
}

void gboost_model_t::prototypes(rwlearners_t&& prototypes)
{
    m_prototypes = std::move(prototypes);
}

std::istream& gboost_model_t::read(std::istream& stream)
{
    learner_t::read(stream);

    critical(::nano::read(stream, m_bias) && ::nano::read(stream, m_wlearners) && ::nano::read(stream, m_prototypes),
             "gboost: failed to read from stream!");

    return stream;
}

std::ostream& gboost_model_t::write(std::ostream& stream) const
{
    learner_t::write(stream);

    critical(::nano::write(stream, m_bias) && ::nano::write(stream, m_wlearners) && ::nano::write(stream, m_prototypes),
             "gboost: failed to write to stream!");

    return stream;
}

ml::result_t gboost_model_t::fit(const dataset_t& dataset, const indices_t& samples, const loss_t& loss,
                                 const ml::params_t& fit_params)
{
    critical(!m_prototypes.empty(), "gboost: cannot fit without any weak learner!");

    // tune hyper-parameters (if any)
    const auto callback = [&](const indices_t& train_samples, const indices_t& valid_samples,
                              const tensor1d_cmap_t params, const std::any&, const logger_t& logger)
    {
        auto [gboost, train_errors_losses, valid_errors_losses] = ::fit(
            *this, dataset, train_samples, valid_samples, loss, fit_params.solver(), m_prototypes, params, logger);

        return std::make_tuple(std::move(train_errors_losses), std::move(valid_errors_losses), std::move(gboost));
    };

    auto fit_result = ml::tune("gboost", samples, fit_params, ::make_params(*this), callback);

    // choose the optimum hyper-parameters and merge the boosters fitted for each fold
    {
        const auto optimum_trial = fit_result.optimum_trial();
        const auto folds         = fit_result.folds();

        m_bias = make_full_tensor<scalar_t>(make_dims(::nano::size(dataset.target_dims())), 0.0);
        m_wlearners.clear();

        for (tensor_size_t fold = 0; fold < folds; ++fold)
        {
            const auto* const pgboost = std::any_cast<gboost::result_t>(&fit_result.extra(optimum_trial, fold));
            assert(pgboost != nullptr);
            m_bias.vector() += pgboost->m_bias.vector();
            std::for_each(pgboost->m_wlearners.begin(), pgboost->m_wlearners.end(),
                          [&](const auto& wlearner) { m_wlearners.emplace_back(wlearner->clone()); });
        }

        ::nano::wlearner::merge(m_wlearners);

        const auto denom  = 1.0 / static_cast<scalar_t>(folds);
        const auto vdenom = make_vector<scalar_t>(denom);
        m_bias.vector() *= denom;
        for (const auto& wlearner : m_wlearners)
        {
            wlearner->scale(vdenom);
        }

        learner_t::fit_dataset(dataset);

        const auto all_samples = arange(0, dataset.samples());
        const auto outputs     = predict(dataset, all_samples);
        auto       values      = tensor2d_t{2, all_samples.size()};
        const auto batch       = parameter("gboost::batch").value<tensor_size_t>();

        auto targets_iterator = targets_iterator_t{dataset, all_samples};
        targets_iterator.batch(batch);
        targets_iterator.scaling(scaling_type::none);
        ::nano::gboost::evaluate(targets_iterator, loss, outputs, values);

        fit_result.store(::selected(values, samples));
    }
    fit_params.log(fit_result, fit_result.trials(), "gboost");

    return fit_result;
}

void gboost_model_t::do_predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const
{
    outputs.reshape(samples.size(), -1).matrix().rowwise() = m_bias.vector().transpose();

    for (const auto& wlearner : m_wlearners)
    {
        wlearner->predict(dataset, samples, outputs.tensor());
    }
}

indices_t gboost_model_t::features() const
{
    std::set<tensor_size_t> ufeatures;
    for (const auto& wlearner : m_wlearners)
    {
        for (const auto feature : wlearner->features())
        {
            ufeatures.emplace(feature);
        }
    }

    indices_t features(static_cast<tensor_size_t>(ufeatures.size()));
    std::copy(ufeatures.begin(), ufeatures.end(), features.begin());

    return features;
}
