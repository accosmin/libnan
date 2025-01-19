#include <nano/core/numeric.h>
#include <nano/dataset.h>
#include <nano/dataset/stats.h>

using namespace nano;

namespace
{
template <class tvalues>
void nan2zero(tvalues& values)
{
    // NB: replace missing scalar values (represented as NaN) with zeros,
    // to train and evaluate dense ML models (e.g. linear models).
    for (tensor_size_t i = 0, size = values.size(); i < size; ++i)
    {
        auto& value = values(i);
        if (!std::isfinite(value))
        {
            value = 0.0;
        }
    }
}

template <class toperator>
auto make_features(const dataset_t& dataset, const toperator& op)
{
    tensor_size_t count = 0;
    for (tensor_size_t i = 0, size = dataset.features(); i < size; ++i)
    {
        if (op(dataset.feature(i)))
        {
            ++count;
        }
    }

    indices_t features(count);
    for (tensor_size_t i = 0, k = 0, size = dataset.features(); i < size; ++i)
    {
        if (op(dataset.feature(i)))
        {
            features(k++) = i;
        }
    }

    return features;
} // LCOV_EXCL_LINE

auto make_scaling(const scalar_stats_t& stats, const scaling_type scaling)
{
    auto w = make_full_tensor<scalar_t>(make_dims(stats.m_min.size()), 1.0);
    auto b = make_full_tensor<scalar_t>(make_dims(stats.m_min.size()), 0.0);

    // NB: check that scalar statistics are initialized!
    if (stats.m_min.size() > 0)
    {
        switch (scaling)
        {
        case scaling_type::mean:
            w         = stats.m_div_range;
            b.array() = -stats.m_mean.array() * stats.m_div_range.array();
            break;

        case scaling_type::minmax:
            w         = stats.m_div_range;
            b.array() = -stats.m_min.array() * stats.m_div_range.array();
            break;

        case scaling_type::standard:
            w         = stats.m_div_stdev;
            b.array() = -stats.m_mean.array() * stats.m_div_stdev.array();
            break;

        default: break;
        }
    }

    return std::make_pair(w, b);
}

void update(scalar_stats_t& stats, const tensor2d_cmap_t& values)
{
    assert(values.size<1>() == stats.m_min.size());

    for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
    {
        for (tensor_size_t column = 0, columns = values.size<1>(); column < columns; ++column)
        {
            const auto value = values(sample, column);
            if (std::isfinite(value))
            {
                stats.m_samples(column) += 1;
                stats.m_mean(column) += value;
                stats.m_stdev(column) += value * value;
                stats.m_min(column) = std::min(stats.m_min(column), value);
                stats.m_max(column) = std::max(stats.m_max(column), value);
            }
        }
    }
}

void done(scalar_stats_t& stats, const tensor_mem_t<uint8_t, 1>& enable_scaling = {})
{
    const auto epsilon = epsilon2<scalar_t>();

    for (tensor_size_t i = 0, size = stats.m_samples.size(); i < size; ++i)
    {
        if (const auto N = stats.m_samples(i); N > 1)
        {
            const auto dN    = static_cast<scalar_t>(N);
            stats.m_stdev(i) = std::sqrt((stats.m_stdev(i) - stats.m_mean(i) * stats.m_mean(i) / dN) / (dN - 1.0));
            stats.m_mean(i) /= dN;
            stats.m_div_range(i) = 1.0 / std::max(stats.m_max(i) - stats.m_min(i), epsilon);
            stats.m_div_stdev(i) = 1.0 / std::max(stats.m_stdev(i), epsilon);
            stats.m_mul_range(i) = std::max(stats.m_max(i) - stats.m_min(i), epsilon);
            stats.m_mul_stdev(i) = std::max(stats.m_stdev(i), epsilon);
        }
        else
        {
            if (N == 0)
            {
                stats.m_min(i)  = 0.0;
                stats.m_max(i)  = 0.0;
                stats.m_mean(i) = 0.0;
            }
            stats.m_stdev(i)     = 0.0;
            stats.m_div_range(i) = 1.0;
            stats.m_div_stdev(i) = 1.0;
            stats.m_mul_range(i) = 1.0;
            stats.m_mul_stdev(i) = 1.0;
        }

        // NB: disable scaling for this dimension!
        if (i < enable_scaling.size() && enable_scaling(i) == 0x00)
        {
            stats.m_min(i)       = 0.0;
            stats.m_max(i)       = 0.0;
            stats.m_mean(i)      = 0.0;
            stats.m_stdev(i)     = 0.0;
            stats.m_div_range(i) = 1.0;
            stats.m_div_stdev(i) = 1.0;
            stats.m_mul_range(i) = 1.0;
            stats.m_mul_stdev(i) = 1.0;
        }
    }
}

template <class tvalues>
auto alloc_xclass_stats(const tvalues& values)
{
    xclass_stats_t stats;
    stats.m_class_hashes  = ::nano::make_hashes(values);
    stats.m_class_samples = make_full_tensor<tensor_size_t>(make_dims(stats.m_class_hashes.size()), 0);
    stats.m_sample_classes.resize(values.template size<0>());
    stats.m_sample_weights.resize(values.template size<0>());
    return stats;
} // LCOV_EXCL_LINE

template <class tvalues>
void update(xclass_stats_t& stats, const tensor_size_t sample, const tvalues& values)
{
    const auto iclass = ::nano::find(stats.m_class_hashes, values);
    if (iclass >= 0)
    {
        stats.m_class_samples(iclass) += 1;
    }
    stats.m_sample_classes(sample) = iclass;
}

void done(xclass_stats_t& stats)
{
    const auto norm = 1.0 / (1.0 / stats.m_class_samples.array().cast<scalar_t>()).sum();

    for (tensor_size_t sample = 0, samples = stats.m_sample_classes.size(); sample < samples; ++sample)
    {
        const auto iclass = stats.m_sample_classes(sample);
        if (iclass >= 0)
        {
            stats.m_sample_weights(sample) = norm / static_cast<scalar_t>(stats.m_class_samples(iclass));
        }
        else
        {
            stats.m_sample_weights(sample) = 0.0;
        }
    }
}

auto make_xclass_stats(const sclass_cmap_t& values)
{
    auto stats = ::alloc_xclass_stats(values);
    for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
    {
        ::update(stats, sample, values(sample));
    }
    ::done(stats);
    return stats;
}

auto make_xclass_stats(const mclass_cmap_t& values)
{
    auto stats = ::alloc_xclass_stats(values);
    for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
    {
        ::update(stats, sample, values.array(sample));
    }
    ::done(stats);
    return stats;
}
} // namespace

void nano::upscale(const scalar_stats_t& flatten_stats, scaling_type flatten_scaling,
                   const scalar_stats_t& targets_stats, scaling_type targets_scaling, tensor2d_map_t weights,
                   tensor1d_map_t bias)
{
    assert(bias.size<0>() == targets_stats.m_min.size());
    assert(weights.size<0>() == targets_stats.m_min.size());
    assert(weights.size<1>() == flatten_stats.m_min.size());

    const auto [flatten_w, flatten_b] = make_scaling(flatten_stats, flatten_scaling);
    const auto [targets_w, targets_b] = make_scaling(targets_stats, targets_scaling);

    // cppcheck-suppress unreadVariable
    bias.array() = (weights.matrix() * flatten_b.vector()).array() + bias.array() - targets_b.array();
    // cppcheck-suppress unreadVariable
    bias.array() /= targets_w.array();

    // cppcheck-suppress unreadVariable
    weights.matrix().array().colwise() /= targets_w.array();
    // cppcheck-suppress unreadVariable
    weights.matrix().array().rowwise() *= flatten_w.array().transpose();
}

indices_t nano::make_sclass_features(const dataset_t& dataset)
{
    return make_features(dataset, [](const auto& feature) { return feature.is_sclass(); });
}

indices_t nano::make_mclass_features(const dataset_t& dataset)
{
    return make_features(dataset, [](const auto& feature) { return feature.is_mclass(); });
}

indices_t nano::make_scalar_features(const dataset_t& dataset)
{
    return make_features(dataset, [](const auto& feature) { return feature.is_scalar(); });
}

indices_t nano::make_struct_features(const dataset_t& dataset)
{
    return make_features(dataset, [](const auto& feature) { return feature.is_struct(); });
}

scalar_stats_t::scalar_stats_t(const tensor_size_t dims)
    : m_samples(make_full_tensor<tensor_size_t>(make_dims(dims), 0))
    , m_min(make_full_tensor<scalar_t>(make_dims(dims), std::numeric_limits<scalar_t>::max()))
    , m_max(make_full_tensor<scalar_t>(make_dims(dims), std::numeric_limits<scalar_t>::lowest()))
    , m_mean(make_full_tensor<scalar_t>(make_dims(dims), 0.0))
    , m_stdev(make_full_tensor<scalar_t>(make_dims(dims), 0.0))
    , m_div_range(make_full_tensor<scalar_t>(make_dims(dims), 1.0))
    , m_mul_range(make_full_tensor<scalar_t>(make_dims(dims), 1.0))
    , m_div_stdev(make_full_tensor<scalar_t>(make_dims(dims), 1.0))
    , m_mul_stdev(make_full_tensor<scalar_t>(make_dims(dims), 1.0))
{
}

scalar_stats_t scalar_stats_t::make_flatten_stats(const dataset_t& dataset, indices_cmap_t samples,
                                                  const tensor_size_t batch)
{
    scalar_stats_t stats{dataset.columns()};

    tensor2d_t buffer;
    for (tensor_size_t i = 0, size = samples.size(); i < size; i += batch)
    {
        const auto range  = make_range(i, std::min(i + batch, size));
        const auto values = dataset.flatten(samples.slice(range), buffer);
        ::update(stats, values);
    }

    auto enable_scaling = tensor_mem_t<uint8_t, 1>(stats.m_min.size());
    for (tensor_size_t column = 0; column < enable_scaling.size(); ++column)
    {
        const auto ifeature    = dataset.column2feature(column);
        const auto feature     = dataset.feature(ifeature);
        const auto isclass     = feature.is_sclass() || feature.is_mclass();
        enable_scaling(column) = isclass ? 0x00 : 0x01;
    }

    ::done(stats, enable_scaling);
    return stats;
}

scalar_stats_t scalar_stats_t::make_targets_stats(const dataset_t& dataset, indices_cmap_t samples,
                                                  const tensor_size_t batch)
{
    const auto& target = dataset.target();
    if (!target.valid())
    {
        raise("scalar statistics cannot be computed for targets of unsupervised datasets!");
    }

    scalar_stats_t stats{::nano::size(dataset.target_dims())};

    tensor4d_t buffer;
    for (tensor_size_t i = 0, size = samples.size(); i < size; i += batch)
    {
        const auto range  = make_range(i, std::min(i + batch, size));
        const auto values = dataset.targets(samples.slice(range), buffer);
        ::update(stats, values.reshape(range.size(), -1));
    }

    if (target.is_sclass() || target.is_mclass())
    {
        ::done(stats, make_full_tensor<uint8_t>(make_dims(stats.m_min.size()), 0x00));
    }
    else
    {
        ::done(stats, make_full_tensor<uint8_t>(make_dims(stats.m_min.size()), 0x01));
    }
    return stats;
}

scalar_stats_t scalar_stats_t::make_feature_stats(const dataset_t& dataset, indices_cmap_t samples,
                                                  const tensor_size_t ifeature, const tensor_size_t batch)
{
    const auto feature = dataset.feature(ifeature);

    scalar_stats_t stats{::nano::size(feature.dims())};
    if (feature.is_scalar())
    {
        tensor1d_t buffer;
        for (tensor_size_t i = 0, size = samples.size(); i < size; i += batch)
        {
            const auto range  = make_range(i, std::min(i + batch, size));
            const auto values = dataset.select(samples.slice(range), ifeature, buffer);
            ::update(stats, values.reshape(range.size(), -1));
        }
    }
    else if (feature.is_struct())
    {
        tensor4d_t buffer;
        for (tensor_size_t i = 0, size = samples.size(); i < size; i += batch)
        {
            const auto range  = make_range(i, std::min(i + batch, size));
            const auto values = dataset.select(samples.slice(range), ifeature, buffer);
            ::update(stats, values.reshape(range.size(), -1));
        }
    }
    else
    {
        raise("scalar statistics cannot be computed for categorical feature: ", feature, "!");
    }

    ::done(stats, make_full_tensor<uint8_t>(make_dims(stats.m_min.size()), 0x01));
    return stats;
}

void scalar_stats_t::scale(const scaling_type scaling, tensor2d_map_t values) const
{
    assert(values.size<1>() == m_min.size());

    switch (scaling)
    {
    case scaling_type::none:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            nan2zero(array);
        }
        break;

    case scaling_type::mean:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            array      = (array - m_mean.array()) * m_div_range.array();
            nan2zero(array);
        }
        break;

    case scaling_type::minmax:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            array      = (array - m_min.array()) * m_div_range.array();
            nan2zero(array);
        }
        break;

    case scaling_type::standard:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            array      = (array - m_mean.array()) * m_div_stdev.array();
            nan2zero(array);
        }
        break;

    default: throw std::runtime_error("unhandled scaling type");
    }
}

void scalar_stats_t::scale(scaling_type scaling, tensor4d_map_t values) const
{
    assert(values.size() == m_min.size() * values.size<0>());

    scale(scaling, values.reshape(values.size<0>(), -1));
}

void scalar_stats_t::upscale(scaling_type scaling, tensor2d_map_t values) const
{
    assert(values.size<1>() == m_min.size());

    switch (scaling)
    {
    case scaling_type::none: break;

    case scaling_type::mean:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            array      = m_mean.array() + array * m_mul_range.array();
        }
        break;

    case scaling_type::minmax:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            array      = m_min.array() + array * m_mul_range.array();
        }
        break;

    case scaling_type::standard:
        for (tensor_size_t sample = 0, samples = values.size<0>(); sample < samples; ++sample)
        {
            auto array = values.array(sample);
            array      = m_mean.array() + array * m_mul_stdev.array();
        }
        break;

    default: throw std::runtime_error("unhandled scaling type");
    }
}

void scalar_stats_t::upscale(scaling_type scaling, tensor4d_map_t values) const
{
    assert(values.size() == m_min.size() * values.size<0>());

    upscale(scaling, values.reshape(values.size<0>(), -1));
}

xclass_stats_t xclass_stats_t::make_targets_stats(const dataset_t& dataset, indices_cmap_t samples)
{
    const auto& target = dataset.target();

    if (target.is_sclass())
    {
        sclass_mem_t buffer;
        return ::make_xclass_stats(dataset.select(samples, buffer));
    }
    else if (target.is_mclass())
    {
        mclass_mem_t buffer;
        return ::make_xclass_stats(dataset.select(samples, buffer));
    }
    else
    {
        raise("class statstics cannot be computed for continuous target: ", target, "!");
    }
}

xclass_stats_t xclass_stats_t::make_feature_stats(const dataset_t& dataset, indices_cmap_t samples,
                                                  const tensor_size_t ifeature)
{
    const auto feature = dataset.feature(ifeature);

    if (feature.is_sclass())
    {
        sclass_mem_t buffer;
        return ::make_xclass_stats(dataset.select(samples, ifeature, buffer));
    }
    else if (feature.is_mclass())
    {
        mclass_mem_t buffer;
        return ::make_xclass_stats(dataset.select(samples, ifeature, buffer));
    }
    else
    {
        raise("class statstics cannot be computed for continuous feature: ", feature, "!");
    }
}
