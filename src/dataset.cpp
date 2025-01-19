#include <nano/core/chrono.h>
#include <nano/dataset.h>

using namespace nano;

namespace
{
void handle_sclass(const feature_t& feature)
{
    critical(feature.is_sclass(), "dataset: unhandled single-label target <", feature, ">!");
}

void handle_mclass(const feature_t& feature)
{
    critical(feature.is_mclass(), "dataset: unhandled multi-label target <", feature, ">!");
}

void handle_scalar(const feature_t& feature)
{
    critical(feature.is_scalar(), "dataset: unhandled scalar target <", feature, ">!");
}

void handle_struct(const feature_t& feature)
{
    critical(feature.is_struct(), "dataset: unhandled structured target <", feature, ">!");
}

void handle_sclass(const tensor_size_t ifeature, const feature_t& feature)
{
    critical(feature.is_sclass(), "dataset: unhandled single-label feature <", ifeature, ":", feature, ">!");
}

void handle_mclass(const tensor_size_t ifeature, const feature_t& feature)
{
    critical(feature.is_mclass(), "dataset: unhandled multi-label feature <", ifeature, ":", feature, ">!");
}

void handle_scalar(const tensor_size_t ifeature, const feature_t& feature)
{
    critical(feature.is_scalar(), "dataset: unhandled scalar feature <", ifeature, ":", feature, ">!");
}

void handle_struct(const tensor_size_t ifeature, const feature_t& feature)
{
    critical(feature.is_struct(), "dataset: unhandled structured feature <", ifeature, ":", feature, ">!");
}

template <class tscalar, size_t trank, class... tindices>
auto resize_and_map(tensor_mem_t<tscalar, trank>& buffer, tindices... dims)
{
    if (buffer.size() < ::nano::size(make_dims(dims...)))
    {
        buffer.resize(dims...);
    }
    return map_tensor(buffer.data(), dims...);
}
} // namespace

dataset_t::dataset_t(const datasource_t& datasource, const size_t threads)
    : m_datasource(datasource)
    , m_pool(std::make_unique<parallel::pool_t>(threads))
{
    if (m_datasource.type() != task_type::unsupervised)
    {
        m_target =
            m_datasource.visit_target([](const feature_t& feature, const auto&, const auto&) { return feature; });
    }
}

dataset_t& dataset_t::add(rgenerator_t&& generator)
{
    const auto timer = ::nano::timer_t{};
    const auto genid = generator->type_id();

    generator->fit(m_datasource);
    m_generators.emplace_back(std::move(generator));
    update();

    const auto elapsed = timer.elapsed();
    log(log_type::info, "dataset: loaded feature generator '", genid, "' in <", elapsed, ">.\n");
    log(log_type::info, "dataset: > columns=", columns(), "\n");
    log(log_type::info, "dataset: > target=[", target(), "]\n");
    return *this;
}

void dataset_t::update()
{
    tensor_size_t features      = 0;
    tensor_size_t generators    = 0;
    tensor_size_t total_columns = 0;
    for (const auto& generator : m_generators)
    {
        for (tensor_size_t ifeature = 0; ifeature < generator->features(); ++ifeature, ++features)
        {
            switch (const auto feature = generator->feature(ifeature); feature.type())
            {
            case feature_type::sclass: total_columns += feature.classes() - 1; break;
            case feature_type::mclass: total_columns += feature.classes(); break;
            default: total_columns += size(feature.dims()); break;
            }
        }
        ++generators;
    }

    m_column_mapping.resize(total_columns, 3);
    m_feature_mapping.resize(features, 5);
    m_generator_mapping.resize(generators, 1);

    tensor_size_t index           = 0;
    tensor_size_t offset_columns  = 0;
    tensor_size_t offset_features = 0;
    for (const auto& generator : m_generators)
    {
        const auto old_offset_columns = offset_columns;

        for (tensor_size_t ifeature = 0; ifeature < generator->features(); ++ifeature, ++offset_features)
        {
            m_feature_mapping(offset_features, 0) = index;
            m_feature_mapping(offset_features, 1) = ifeature;

            tensor_size_t dim1    = 1;
            tensor_size_t dim2    = 1;
            tensor_size_t dim3    = 1;
            tensor_size_t columns = 0;
            switch (const auto feature = generator->feature(ifeature); feature.type())
            {
            case feature_type::sclass: columns = feature.classes() - 1; break;
            case feature_type::mclass:
                dim1    = feature.classes();
                columns = feature.classes();
                break;
            default:
                dim1    = feature.dims()[0];
                dim2    = feature.dims()[1];
                dim3    = feature.dims()[2];
                columns = size(feature.dims());
                break;
            }
            m_feature_mapping(offset_features, 2) = dim1;
            m_feature_mapping(offset_features, 3) = dim2;
            m_feature_mapping(offset_features, 4) = dim3;

            for (tensor_size_t icolumn = 0; icolumn < columns; ++icolumn, ++offset_columns)
            {
                m_column_mapping(offset_columns, 0) = index;
                m_column_mapping(offset_columns, 1) = icolumn;
                m_column_mapping(offset_columns, 2) = offset_features;
            }
        }

        m_generator_mapping(index++, 0) = offset_columns - old_offset_columns;
    }
}

tensor_size_t dataset_t::features() const
{
    return m_feature_mapping.size<0>();
}

feature_t dataset_t::feature(tensor_size_t feature) const
{
    return byfeature(feature)->feature(m_feature_mapping(feature, 1));
}

tensor_size_t dataset_t::columns() const
{
    return m_column_mapping.size<0>();
}

tensor_size_t dataset_t::column2feature(tensor_size_t column) const
{
    return m_column_mapping(column, 2);
}

sclass_cmap_t dataset_t::select(indices_cmap_t samples, sclass_mem_t& buffer) const
{
    check(samples);
    handle_sclass(m_target);

    return m_datasource.visit_target(
        [&](const feature_t&, const auto& data, const auto& mask)
        {
            auto storage = resize_and_map(buffer, samples.size());
            loop_samples(
                data, mask, samples, indices_cmap_t{},
                [&](auto it)
                {
                    for (; it; ++it)
                    {
                        if (const auto [index, given, label] = *it; given)
                        {
                            storage(index) = static_cast<int32_t>(label);
                        }
                        else
                        {
                            storage(index) = -1;
                        }
                    }
                },
                [&](auto) {}, [&](auto) {});
            return storage;
        });
}

mclass_cmap_t dataset_t::select(indices_cmap_t samples, mclass_mem_t& buffer) const
{
    check(samples);
    handle_mclass(m_target);

    return m_datasource.visit_target(
        [&](const feature_t& feature, const auto& data, const auto& mask)
        {
            auto storage = resize_and_map(buffer, samples.size(), feature.classes());
            loop_samples(
                data, mask, samples, indices_cmap_t{}, [&](auto) {},
                [&](auto it)
                {
                    for (; it; ++it)
                    {
                        if (const auto [index, given, hits] = *it; given)
                        {
                            storage.array(index) = hits.array().template cast<int8_t>();
                        }
                        else
                        {
                            storage.array(index) = -1;
                        }
                    }
                },
                [&](auto) {});
            return storage;
        });
}

scalar_cmap_t dataset_t::select(indices_cmap_t samples, scalar_mem_t& buffer) const
{
    check(samples);
    handle_scalar(m_target);

    return m_datasource.visit_target(
        [&](const feature_t&, const auto& data, const auto& mask)
        {
            auto storage = resize_and_map(buffer, samples.size());
            loop_samples(
                data, mask, samples, indices_cmap_t{}, [&](auto) {}, [&](auto) {},
                [&](auto it)
                {
                    for (; it; ++it)
                    {
                        if (const auto [index, given, values] = *it; given)
                        {
                            storage(index) = static_cast<scalar_t>(values(0));
                        }
                        else
                        {
                            storage(index) = std::numeric_limits<scalar_t>::quiet_NaN();
                        }
                    }
                });
            return storage;
        });
}

struct_cmap_t dataset_t::select(indices_cmap_t samples, struct_mem_t& buffer) const
{
    check(samples);
    handle_struct(m_target);

    return m_datasource.visit_target(
        [&](const feature_t& feature, const auto& data, const auto& mask)
        {
            const auto [dim1, dim2, dim3] = feature.dims();

            auto storage = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
            loop_samples(
                data, mask, samples, indices_cmap_t{}, [&](auto) {}, [&](auto) {},
                [&](auto it)
                {
                    for (; it; ++it)
                    {
                        if (const auto [index, given, values] = *it; given)
                        {
                            storage.array(index) = values.array().template cast<scalar_t>();
                        }
                        else
                        {
                            storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                    }
                });
            return storage;
        });
}

sclass_cmap_t dataset_t::select(indices_cmap_t samples, tensor_size_t feature, sclass_mem_t& buffer) const
{
    check(samples);
    handle_sclass(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size());
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

mclass_cmap_t dataset_t::select(indices_cmap_t samples, tensor_size_t feature, mclass_mem_t& buffer) const
{
    check(samples);
    handle_mclass(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size(), m_feature_mapping(feature, 2));
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

scalar_cmap_t dataset_t::select(indices_cmap_t samples, tensor_size_t feature, scalar_mem_t& buffer) const
{
    check(samples);
    handle_scalar(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size());
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

struct_cmap_t dataset_t::select(indices_cmap_t samples, tensor_size_t feature, struct_mem_t& buffer) const
{
    check(samples);
    handle_struct(feature, this->feature(feature));

    auto storage = resize_and_map(buffer, samples.size(), m_feature_mapping(feature, 2), m_feature_mapping(feature, 3),
                                  m_feature_mapping(feature, 4));
    byfeature(feature)->select(samples, m_feature_mapping(feature, 1), storage);
    return storage;
}

tensor2d_map_t dataset_t::flatten(indices_cmap_t samples, tensor2d_t& buffer) const
{
    check(samples);

    const auto storage = resize_and_map(buffer, samples.size(), columns());

    tensor_size_t index  = 0;
    tensor_size_t offset = 0;
    for (const auto& generator : m_generators)
    {
        generator->flatten(samples, storage, offset);
        offset += m_generator_mapping(index++, 0);
    }

    return storage;
}

tensor3d_dims_t dataset_t::target_dims() const
{
    switch (m_datasource.type())
    {
    case task_type::unsupervised: return make_dims(0, 0, 0);

    default:
        return m_datasource.visit_target(
            [](const feature_t& feature, const auto&, const auto&)
            {
                switch (feature.type())
                {
                case feature_type::sclass: return make_dims(feature.classes(), 1, 1); // NOLINT(bugprone-branch-clone)
                case feature_type::mclass: return make_dims(feature.classes(), 1, 1);
                default: return feature.dims();
                }
            });
    }
}

tensor4d_map_t dataset_t::targets(indices_cmap_t samples, tensor4d_t& buffer) const
{
    check(samples);

    if (m_datasource.type() == task_type::unsupervised)
    {
        raise("dataset: targets are not available for unsupervised datasets!");
    }

    return m_datasource.visit_target(
        [&](const feature_t& feature, const auto& data, const auto& mask)
        {
            return loop_samples(
                data, mask, samples, indices_cmap_t{},
                [&](auto it)
                {
                    const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
                    for (; it; ++it)
                    {
                        if (const auto [index, given, label] = *it; given)
                        {
                            storage.array(index).setConstant(-1.0);
                            storage.array(index)(static_cast<tensor_size_t>(label)) = +1.0;
                        }
                        else
                        {
                            storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                    }
                    return tensor4d_map_t{storage};
                },
                [&](auto it)
                {
                    const auto storage = resize_and_map(buffer, samples.size(), feature.classes(), 1, 1);
                    for (; it; ++it)
                    {
                        if (const auto [index, given, hits] = *it; given)
                        {
                            storage.array(index) = hits.array().template cast<scalar_t>() * 2.0 - 1.0;
                        }
                        else
                        {
                            storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                    }
                    return tensor4d_map_t{storage};
                },
                [&](auto it)
                {
                    const auto [dim1, dim2, dim3] = feature.dims();
                    const auto storage            = resize_and_map(buffer, samples.size(), dim1, dim2, dim3);
                    for (; it; ++it)
                    {
                        if (const auto [index, given, values] = *it; given)
                        {
                            storage.array(index) = values.array().template cast<scalar_t>();
                        }
                        else
                        {
                            storage.array(index).setConstant(std::numeric_limits<scalar_t>::quiet_NaN());
                        }
                    }
                    return tensor4d_map_t{storage};
                });
        });
}

void dataset_t::undrop() const
{
    for (const auto& generator : m_generators)
    {
        generator->undrop();
    }
}

void dataset_t::drop(const tensor_size_t feature) const
{
    byfeature(feature)->drop(m_feature_mapping(feature, 1));
}

void dataset_t::unshuffle() const
{
    for (const auto& generator : m_generators)
    {
        generator->unshuffle();
    }
}

void dataset_t::shuffle(const tensor_size_t feature) const
{
    byfeature(feature)->shuffle(m_feature_mapping(feature, 1));
}

indices_t dataset_t::shuffled(const tensor_size_t feature, indices_cmap_t samples) const
{
    return byfeature(feature)->shuffled(m_feature_mapping(feature, 1), samples);
}

const rgenerator_t& dataset_t::byfeature(const tensor_size_t feature) const
{
    check(feature);

    return m_generators[static_cast<size_t>(m_feature_mapping(feature, 0))];
}

void dataset_t::check(tensor_size_t feature) const
{
    critical(feature >= 0 && feature < features(), "dataset: invalid feature index, expecting in [0, ", features(),
             "), got ", feature, "!");
}

void dataset_t::check(indices_cmap_t samples) const
{
    critical(samples.min() >= 0 && samples.max() < m_datasource.samples(),
             "dataset: invalid sample range, expecting in [0, ", m_datasource.samples(), "), got ", "[", samples.min(),
             ", ", samples.max(), ")!");
}
