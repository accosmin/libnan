#include <nano/core/stream.h>
#include <nano/dataset/iterator.h>
#include <nano/learner.h>
#include <nano/loss.h>

using namespace nano;

learner_t::learner_t() = default;

void learner_t::critical_compatible(const dataset_t& dataset) const
{
    const auto n_features = dataset.features();

    critical(n_features == static_cast<tensor_size_t>(m_inputs.size()), "learner: mis-matching number of inputs (",
             n_features, "), expecting (", m_inputs.size(), ")!");

    for (tensor_size_t i = 0; i < n_features; ++i)
    {
        const auto  feature          = dataset.feature(i);
        const auto& expected_feature = m_inputs[static_cast<size_t>(i)];

        critical(feature == expected_feature, "learner: mis-matching input [", i, "/", n_features, "] (", feature,
                 "), expecting (", expected_feature, ")!");
    }

    critical(dataset.target() == m_target, "learner: mis-matching target (", dataset.target(), "), expecting (",
             m_target, ")!");
}

std::istream& learner_t::read(std::istream& stream)
{
    configurable_t::read(stream);

    critical(::nano::read(stream, m_inputs) && ::nano::read(stream, m_target), "learner: failed to read from stream!");

    return stream;
}

std::ostream& learner_t::write(std::ostream& stream) const
{
    configurable_t::write(stream);

    critical(::nano::write(stream, m_inputs) && ::nano::write(stream, m_target), "learner: failed to write to stream!");

    return stream;
}

void learner_t::fit_dataset(const dataset_t& dataset)
{
    const auto n_features = dataset.features();

    m_target = dataset.target();
    m_inputs.clear();
    m_inputs.reserve(static_cast<size_t>(n_features));
    for (tensor_size_t i = 0; i < n_features; ++i)
    {
        m_inputs.push_back(dataset.feature(i));
    }
}

tensor4d_t learner_t::predict(const dataset_t& dataset, indices_cmap_t samples) const
{
    tensor4d_t outputs(cat_dims(samples.size(), dataset.target_dims()));
    outputs.zero();
    predict(dataset, samples, outputs);

    return outputs;
}

void learner_t::predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const
{
    critical_compatible(dataset);

    assert(outputs.dims() == cat_dims(samples.size(), dataset.target_dims()));

    do_predict(dataset, samples, outputs);
}

tensor2d_t learner_t::evaluate(const dataset_t& dataset, indices_cmap_t samples, const loss_t& loss) const
{
    auto errors_values = tensor2d_t{2, samples.size()};

    const auto iterator = targets_iterator_t{dataset, samples};
    iterator.loop(
        [&](const tensor_range_t& range, const size_t, const tensor4d_cmap_t targets)
        {
            const auto outputs = predict(dataset, samples.slice(range));
            loss.error(targets, outputs, errors_values.tensor(0).slice(range));
            loss.value(targets, outputs, errors_values.tensor(1).slice(range));
        });

    return errors_values;
}
