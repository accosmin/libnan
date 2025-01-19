#include <mutex>
#include <nano/generator/elemwise_gradient.h>
#include <nano/generator/elemwise_identity.h>
#include <nano/generator/pairwise_product.h>

using namespace nano;

generator_t::generator_t(string_t id)
    : typed_t(std::move(id))
{
}

void generator_t::fit(const datasource_t& datasource)
{
    m_datasource = &datasource;
}

void generator_t::allocate(const tensor_size_t features)
{
    m_feature_infos.resize(features);
    m_feature_infos.zero();
}

void generator_t::undrop()
{
    m_feature_infos.array() = 0x00;
}

void generator_t::drop(const tensor_size_t feature)
{
    m_feature_infos(feature) = 0x01;
}

void generator_t::unshuffle()
{
    m_feature_infos.array() = 0x00;
    m_feature_shuffles.clear();
}

void generator_t::shuffle(const tensor_size_t feature)
{
    m_feature_infos(feature) = 0x02;

    auto rng = make_rng();

    indices_t shuffled = arange(0, datasource().samples());
    std::shuffle(std::begin(shuffled), std::end(shuffled), rng);
    m_feature_shuffles[feature] = shuffled;
}

indices_t generator_t::shuffled(const tensor_size_t feature, indices_cmap_t samples) const
{
    const auto shuffled_all_samples = shuffled(feature);

    auto shuffled = indices_t{samples.size()};
    for (tensor_size_t i = 0; i < samples.size(); ++i)
    {
        assert(samples(i) >= 0 && samples(i) < shuffled_all_samples.size());
        shuffled(i) = shuffled_all_samples(samples(i));
    }

    return shuffled;
}

bool generator_t::should_drop(const tensor_size_t feature) const
{
    return m_feature_infos(feature) == 0x01;
}

indices_cmap_t generator_t::shuffled(const tensor_size_t feature) const
{
    if (m_feature_infos(feature) == 0x02)
    {
        const auto it = m_feature_shuffles.find(feature);
        assert(it != m_feature_shuffles.end());
        return it->second;
    }
    else
    {
        return indices_cmap_t{};
    }
}

void generator_t::flatten_dropped(tensor2d_map_t storage, const tensor_size_t column, const tensor_size_t colsize)
{
    const auto samples                                          = storage.size<0>();
    storage.matrix().block(0, column, samples, colsize).array() = NaN;
}

const datasource_t& generator_t::datasource() const
{
    critical(m_datasource != nullptr, "generator: cannot access the dataset before fitting!");

    return *m_datasource;
}

void generator_t::select(indices_cmap_t samples, const tensor_size_t ifeature, scalar_map_t storage) const
{
    if (should_drop(ifeature))
    {
        storage.full(NaN);
    }
    else
    {
        do_select(samples, ifeature, storage);
    }
}

void generator_t::select(indices_cmap_t samples, const tensor_size_t ifeature, sclass_map_t storage) const
{
    if (should_drop(ifeature))
    {
        storage.full(-1);
    }
    else
    {
        do_select(samples, ifeature, storage);
    }
}

void generator_t::select(indices_cmap_t samples, const tensor_size_t ifeature, mclass_map_t storage) const
{
    if (should_drop(ifeature))
    {
        storage.full(-1);
    }
    else
    {
        do_select(samples, ifeature, storage);
    }
}

void generator_t::select(indices_cmap_t samples, const tensor_size_t ifeature, struct_map_t storage) const
{
    if (should_drop(ifeature))
    {
        storage.full(NaN);
    }
    else
    {
        do_select(samples, ifeature, storage);
    }
}

factory_t<generator_t>& generator_t::all()
{
    static auto manager = factory_t<generator_t>{};
    const auto  op      = []()
    {
        manager.add<gradient_generator_t>(
            "gradient-like features (e.g. edge orientation & magnitude) from structured features (e.g. images)");

        manager.add<sclass_identity_generator_t>("identity transformation, forward the single-label features");
        manager.add<mclass_identity_generator_t>("identity transformation, forward the multi-label features");
        manager.add<scalar_identity_generator_t>("identity transformation, forward the scalar features");
        manager.add<struct_identity_generator_t>(
            "identity transformation, forward the structured features (e.g. images)");

        manager.add<pairwise_product_generator_t>("product of scalar features to generate quadratic terms");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
