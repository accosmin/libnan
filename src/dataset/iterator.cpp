#include <nano/dataset.h>
#include <nano/dataset/iterator.h>

using namespace nano;

base_dataset_iterator_t::base_dataset_iterator_t(const dataset_t& dataset, size_t threads)
    : m_dataset(dataset)
    , m_pool(threads)
{
}

targets_iterator_t::targets_iterator_t(const dataset_t& dataset, indices_cmap_t samples, size_t threads)
    : base_dataset_iterator_t(dataset, threads)
    , m_samples(samples)
    , m_targets_buffers(concurrency())
{
    m_targets_stats = make_targets_stats();
}

targets_stats_t targets_iterator_t::make_targets_stats() const
{
    const auto& datasource = dataset().datasource();
    if (datasource.type() == task_type::unsupervised)
    {
        return targets_stats_t{};
    }
    else
    {
        return datasource.visit_target(
            [&](const auto& feature, const auto& data, const auto& mask)
            {
                return loop_samples(
                    data, mask, m_samples,
                    [&](auto it) -> targets_stats_t { return sclass_stats_t::make(feature, it); },
                    [&](auto it) -> targets_stats_t { return mclass_stats_t::make(feature, it); },
                    [&](auto it) -> targets_stats_t { return scalar_stats_t::make(feature, it); });
            });
    }
}

tensor4d_cmap_t targets_iterator_t::targets(tensor4d_map_t data) const
{
    if (std::holds_alternative<scalar_stats_t>(m_targets_stats))
    {
        std::get<scalar_stats_t>(m_targets_stats).scale(m_scaling, data);
    }
    return data;
}

tensor4d_cmap_t targets_iterator_t::targets(size_t tnum, const tensor_range_t& range) const
{
    if (m_targets.size<0>() == m_samples.size())
    {
        return m_targets.slice(range);
    }
    else
    {
        assert(tnum < m_targets_buffers.size());
        return targets(dataset().targets(m_samples.slice(range), m_targets_buffers[tnum]));
    }
}

bool targets_iterator_t::cache_targets(tensor_size_t max_bytes)
{
    auto       cached = false;
    const auto tdims  = dataset().target_dims();

    if (static_cast<tensor_size_t>(sizeof(scalar_t)) * m_samples.size() * nano::size(tdims) <= max_bytes)
    {
        try
        {
            m_targets.resize(cat_dims(m_samples.size(), tdims));
            map(m_samples.size(), batch(),
                [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
                {
                    assert(tnum < m_targets_buffers.size());
                    const auto range       = make_range(begin, end);
                    const auto samples     = m_samples.slice(range);
                    m_targets.slice(range) = targets(dataset().targets(samples, m_targets_buffers[tnum]));
                });
            cached = true;
        }
        catch (...)
        {
        } // LCOV_EXCL_LINE
    }

    return cached;
}

void targets_iterator_t::batch(tensor_size_t batch)
{
    m_batch = batch;
}

void targets_iterator_t::scaling(scaling_type scaling)
{
    m_scaling = scaling;
}

flatten_iterator_t::flatten_iterator_t(const dataset_t& dataset, indices_cmap_t samples, size_t threads)
    : targets_iterator_t(dataset, samples, threads)
    , m_flatten_buffers(concurrency())
{
    m_flatten_stats = make_flatten_stats();
}

flatten_stats_t flatten_iterator_t::make_flatten_stats()
{
    const auto& samples = this->samples();
    const auto& dataset = this->dataset();

    std::vector<flatten_stats_t> stats(concurrency(), flatten_stats_t{dataset.columns()});
    map(samples.size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            assert(tnum < m_flatten_buffers.size());
            const auto range = make_range(begin, end);
            const auto data  = dataset.flatten(samples.slice(range), m_flatten_buffers[tnum]);
            for (tensor_size_t i = 0, size = range.size(); i < size; ++i)
            {
                stats[tnum] += data.array(i);
            }
        });

    auto enable_scaling = tensor_mem_t<uint8_t, 1>(dataset.columns());
    for (tensor_size_t column = 0; column < enable_scaling.size(); ++column)
    {
        const auto ifeature    = dataset.column2feature(column);
        const auto feature     = dataset.feature(ifeature);
        const auto isclass     = feature.type() == feature_type::sclass || feature.type() == feature_type::mclass;
        enable_scaling(column) = isclass ? 0x00 : 0x01;
    }

    for (size_t i = 1; i < stats.size(); ++i)
    {
        stats[0] += stats[i];
    }
    return stats[0].done(enable_scaling);
}

tensor2d_cmap_t flatten_iterator_t::flatten(tensor2d_map_t data) const
{
    m_flatten_stats.scale(scaling(), data);
    return data;
}

tensor2d_cmap_t flatten_iterator_t::flatten(size_t tnum, const tensor_range_t& range) const
{
    const auto& samples = this->samples();
    const auto& dataset = this->dataset();

    if (m_flatten.size<0>() == samples.size())
    {
        return m_flatten.slice(range);
    }
    else
    {
        assert(tnum < m_flatten_buffers.size());
        return flatten(dataset.flatten(samples.slice(range), m_flatten_buffers[tnum]));
    }
}

bool flatten_iterator_t::cache_flatten(tensor_size_t max_bytes)
{
    const auto& samples = this->samples();
    const auto& dataset = this->dataset();

    auto       cached = false;
    const auto isize  = dataset.columns();

    if (static_cast<tensor_size_t>(sizeof(scalar_t)) * samples.size() * isize <= max_bytes)
    {
        try
        {
            m_flatten.resize(samples.size(), isize);
            map(samples.size(), batch(),
                [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
                {
                    assert(tnum < m_flatten_buffers.size());
                    const auto range       = make_range(begin, end);
                    m_flatten.slice(range) = flatten(dataset.flatten(samples.slice(range), m_flatten_buffers[tnum]));
                });
            cached = true;
        }
        catch (...)
        {
        } // LCOV_EXCL_LINE
    }

    return cached;
}

void flatten_iterator_t::loop(const flatten_targets_callback_t& callback) const
{
    map(samples().size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            const auto range = make_range(begin, end);

            callback(range, tnum, flatten(tnum, range), targets(tnum, range));
        });
}

void flatten_iterator_t::loop(const flatten_callback_t& callback) const
{
    map(samples().size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            const auto range = make_range(begin, end);

            callback(range, tnum, flatten(tnum, range));
        });
}

void targets_iterator_t::loop(const targets_callback_t& callback) const
{
    map(samples().size(), batch(),
        [&](tensor_size_t begin, tensor_size_t end, size_t tnum)
        {
            const auto range = make_range(begin, end);

            callback(range, tnum, targets(tnum, range));
        });
}

select_iterator_t::select_iterator_t(const dataset_t& dataset, size_t threads)
    : base_dataset_iterator_t(dataset, threads)
    , m_buffers(concurrency())
{
}

void select_iterator_t::loop(indices_cmap_t samples, const sclass_callback_t& callback) const
{
    return loop(samples, dataset().sclass_features(), callback);
}

void select_iterator_t::loop(indices_cmap_t samples, const mclass_callback_t& callback) const
{
    return loop(samples, dataset().mclass_features(), callback);
}

void select_iterator_t::loop(indices_cmap_t samples, const scalar_callback_t& callback) const
{
    return loop(samples, dataset().scalar_features(), callback);
}

void select_iterator_t::loop(indices_cmap_t samples, const struct_callback_t& callback) const
{
    return loop(samples, dataset().struct_features(), callback);
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const sclass_callback_t& callback) const
{
    map(features.size(),
        [&](tensor_size_t index, size_t tnum)
        {
            assert(tnum < m_buffers.size());
            const auto ifeature = features(index);
            callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_sclass));
        });
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const mclass_callback_t& callback) const
{
    map(features.size(),
        [&](tensor_size_t index, size_t tnum)
        {
            assert(tnum < m_buffers.size());
            const auto ifeature = features(index);
            callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_mclass));
        });
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const scalar_callback_t& callback) const
{
    map(features.size(),
        [&](tensor_size_t index, size_t tnum)
        {
            assert(tnum < m_buffers.size());
            const auto ifeature = features(index);
            callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_scalar));
        });
}

void select_iterator_t::loop(indices_cmap_t samples, indices_cmap_t features, const struct_callback_t& callback) const
{
    map(features.size(),
        [&](tensor_size_t index, size_t tnum)
        {
            assert(tnum < m_buffers.size());
            const auto ifeature = features(index);
            callback(ifeature, tnum, dataset().select(samples, ifeature, m_buffers[tnum].m_struct));
        });
}