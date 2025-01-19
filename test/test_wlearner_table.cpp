#include <fixture/wlearner.h>
#include <nano/wlearner/affine.h>
#include <nano/wlearner/dtree.h>
#include <nano/wlearner/table.h>

using namespace nano;

namespace
{
auto make_dense_wlearner()
{
    return dense_table_wlearner_t{};
}

auto make_dstep_wlearner()
{
    return dstep_table_wlearner_t{};
}

auto make_kbest_wlearner()
{
    return kbest_table_wlearner_t{};
}

auto make_ksplit_wlearner()
{
    return ksplit_table_wlearner_t{};
}

auto make_hashes_mclass3()
{
    return ::nano::make_hashes(make_tensor<int8_t, 2>(make_dims(8, 3), 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
                                                      0, 1, 1, 1, 0, 1, 1, 1));
}

auto make_dstep_tables(const tensor_size_t classes, const tensor_size_t fv)
{
    auto tables = make_random_tensor<scalar_t>(make_dims(classes, 1, 1, 1), -1e-5, +1e-5);
    tables(fv)  = -0.42 + 0.37 * static_cast<scalar_t>(fv);
    return tables;
}

auto make_dstep_noise(const tensor_size_t classes, const tensor_size_t fv)
{
    auto noise = make_full_tensor<scalar_t>(make_dims(classes), 1e-6);
    noise(fv)  = 1e-10;
    return noise;
}
} // namespace

template <class twlearner>
class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    using maker_t = std::function<twlearner()>;

    fixture_datasource_t(const tensor_size_t samples, const tensor_size_t feature, tensor4d_t tables,
                         tensor4d_t dense_tables, hashes_t hashes, indices_t hash2tables, tensor1d_t noise,
                         maker_t maker)
        : wlearner_datasource_t(samples, tables.size<0>())
        , m_feature(feature)
        , m_tables(std::move(tables))
        , m_dense_tables(std::move(dense_tables))
        , m_hashes(std::move(hashes))
        , m_hash2tables(std::move(hash2tables))
        , m_noise(std::move(noise))
        , m_maker(std::move(maker))
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    auto make_wlearner() const { return m_maker(); }

    auto make_compatible_wlearners() const
    {
        auto wlearner = make_wlearner();

        auto wlearners = rwlearners_t{};
        wlearners.emplace_back(wlearner.clone());
        return wlearners;
    }

    auto make_incompatible_wlearners() const
    {
        auto wlearner = make_wlearner();

        auto wlearners = rwlearners_t{};
        wlearners.emplace_back(affine_wlearner_t{}.clone());
        wlearners.emplace_back(dtree_wlearner_t{}.clone());
        if (wlearner.type_id() == "dense-table")
        {
            wlearners.emplace_back(dstep_table_wlearner_t{}.clone());
            // wlearners.emplace_back(kbest_table_wlearner_t{}.clone());
            // wlearners.emplace_back(ksplit_table_wlearner_t{}.clone());
        }
        else if (wlearner.type_id() == "dstep-table")
        {
            wlearners.emplace_back(dense_table_wlearner_t{}.clone());
            // wlearners.emplace_back(kbest_table_wlearner_t{}.clone());
            wlearners.emplace_back(ksplit_table_wlearner_t{}.clone());
        }
        /*else if (wlearner.type_id() == "kbest-table")
        {
            // wlearners.emplace_back(dense_table_wlearner_t{}.clone());
            // wlearners.emplace_back(dstep_table_wlearner_t{}.clone());
            // wlearners.emplace_back(ksplit_table_wlearner_t{}.clone());
        }
        else if (wlearner.type_id() == "kbest-table")
        {
            // wlearners.emplace_back(dense_table_wlearner_t{}.clone());
            wlearners.emplace_back(dstep_table_wlearner_t{}.clone());
            // wlearners.emplace_back(kbest_table_wlearner_t{}.clone());
        }*/
        return wlearners;
    }

    auto expected_feature() const { return m_feature; }

    auto expected_features() const { return make_indices(expected_feature()); }

    const auto& expected_tables() const { return m_tables; }

    const auto& expected_hashes() const { return m_hashes; }

    const auto& expected_hash2tables() const { return m_hash2tables; }

    const auto& expected_dense_tables() const { return m_dense_tables; }

    void check_wlearner(const table_wlearner_t& wlearner) const
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-10);
        UTEST_CHECK_EQUAL(wlearner.hashes(), expected_hashes());
        UTEST_CHECK_EQUAL(wlearner.hash2tables(), expected_hash2tables());
    }

private:
    auto make_target() const
    {
        return tensor3d_t{m_dense_tables.template size<1>(), m_dense_tables.template size<2>(),
                          m_dense_tables.template size<3>()};
    }

    void add_noise(const tensor_size_t hash, const tensor3d_t& target, tensor3d_t& noisy_target) const
    {
        assert(hash >= 0 && hash < m_noise.size());

        // add per labeling noise (to check k-best and d-step variations)
        noisy_target.random(-m_noise(hash), +m_noise(hash));
        noisy_target.array() += target.array();
    }

    template <class tfvalue>
    auto make_cluster(const tfvalue& fvalue) const
    {
        // map labeling to the right cluster
        auto cluster = find(m_hashes, fvalue);
        if (cluster >= 0)
        {
            cluster = m_hash2tables(cluster);
        }
        return cluster;
    }

    void do_load_sclass()
    {
        const auto feature = expected_feature();
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(this->samples()), tensor_size_t{0}, classes - 1);

        assert(classes == m_dense_tables.template size<0>());

        auto noisy_target = make_target();

        set_targets(feature,
                    [&](const tensor_size_t sample)
                    {
                        auto [fvalue, target, cluster] = make_table_target(fvalues(sample), m_dense_tables);

                        cluster = make_cluster(fvalue);

                        add_noise(fvalue, target, noisy_target);

                        return std::make_tuple(fvalue, noisy_target, cluster);
                    });
    }

    void do_load_mclass()
    {
        const auto feature = expected_feature();
        const auto classes = this->feature(feature).classes();
        const auto fvalues = make_random_tensor<int8_t, 2>(make_dims(this->samples(), classes), 0, 1);

        const auto dense_hashes = make_hashes_mclass3();
        assert(dense_hashes.size() == m_dense_tables.template size<0>());

        auto noisy_target = make_target();

        set_targets(feature,
                    [&](const tensor_size_t sample)
                    {
                        auto [fvalue, target, cluster] =
                            make_table_target(fvalues.tensor(sample), m_dense_tables, dense_hashes);

                        cluster = make_cluster(fvalue);

                        add_noise(find(dense_hashes, fvalue), target, noisy_target);

                        return std::make_tuple(fvalue, noisy_target, cluster);
                    });
    }

    void do_load() override
    {
        random_datasource_t::do_load();

        assert(m_noise.size() == m_dense_tables.size<0>());
        assert(m_hash2tables.size() == m_hashes.size<0>());

        switch (feature(expected_feature()).type())
        {
        case feature_type::sclass: do_load_sclass(); break;

        case feature_type::mclass: do_load_mclass(); break;

        default: assert(false);
        }
    }

    tensor_size_t m_feature{0};
    tensor4d_t    m_tables;
    tensor4d_t    m_dense_tables;
    hashes_t      m_hashes;
    indices_t     m_hash2tables;
    tensor1d_t    m_noise;
    maker_t       m_maker;
};

UTEST_BEGIN_MODULE(test_wlearner_table)

UTEST_CASE(fit_predict_sclass_dense)
{
    using fixture_t = fixture_datasource_t<dense_table_wlearner_t>;

    const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -1.42, +1.42, -0.42);
    const auto hashes      = make_hashes(make_tensor<int32_t>(make_dims(3), 0, 1, 2));
    const auto hash2tables = make_indices(0, 1, 2);
    const auto noise       = make_full_tensor<scalar_t>(make_dims(3), 1e-12);
    const auto maker       = make_dense_wlearner;

    const auto datasource0 = make_datasource<fixture_t>(90, 1, tables, tables, hashes, hash2tables, noise, maker);
    const auto datasourceX = make_random_datasource(make_features_all_continuous());

    check_wlearner(datasource0, datasourceX);
}

UTEST_CASE(fit_predict_mclass_dense)
{
    using fixture_t = fixture_datasource_t<dense_table_wlearner_t>;

    const auto tables      = make_random_tensor<scalar_t>(make_dims(8, 1, 1, 1));
    const auto hashes      = make_hashes_mclass3();
    const auto hash2tables = arange(0, 8);
    const auto noise       = make_full_tensor<scalar_t>(make_dims(8), 1e-12);
    const auto maker       = make_dense_wlearner;

    const auto datasource0 = make_datasource<fixture_t>(150, 3, tables, tables, hashes, hash2tables, noise, maker);
    const auto datasourceX = make_random_datasource(make_features_all_continuous());

    check_wlearner(datasource0, datasourceX);
}

UTEST_CASE(fit_predict_sclass_dstep)
{
    using fixture_t = fixture_datasource_t<dstep_table_wlearner_t>;

    const auto hash2tables = make_indices(0);
    const auto maker       = make_dstep_wlearner;

    for (tensor_size_t fv = 0; fv < 3; ++fv)
    {
        const auto tables = make_dstep_tables(3, fv);
        const auto tablex = tables.slice(fv, fv + 1);
        const auto hashes = make_hashes(make_tensor<int32_t>(make_dims(1), fv));
        const auto noise  = make_dstep_noise(3, fv);

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
}

UTEST_CASE(fit_predict_mclass_dstep)
{
    using fixture_t = fixture_datasource_t<dstep_table_wlearner_t>;

    const auto hash2tables  = make_indices(0);
    const auto maker        = make_dstep_wlearner;
    const auto dense_hashes = make_hashes_mclass3();

    for (tensor_size_t fv = 0; fv < 8; ++fv)
    {
        const auto tables = make_dstep_tables(8, fv);
        const auto tablex = tables.slice(fv, fv + 1);
        const auto hashes = dense_hashes.slice(fv, fv + 1);
        const auto noise  = make_dstep_noise(8, fv);

        const auto datasource0 = make_datasource<fixture_t>(150, 3, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
}

UTEST_CASE(fit_predict_sclass_kbest)
{
    using fixture_t = fixture_datasource_t<kbest_table_wlearner_t>;

    const auto maker = make_kbest_wlearner;
    {
        const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), 0.0, 0.0, -0.42);
        const auto noise       = make_tensor<scalar_t>(make_dims(3), 1e-10, 1e-10, 1e-10);
        const auto hash2tables = make_indices(0);
        const auto tablex      = tables.slice(2, 3);
        const auto hashes      = make_hashes(make_tensor<int32_t>(make_dims(1), 2));

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
    {
        const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), +1.42, 0.0, -0.42);
        const auto noise       = make_tensor<scalar_t>(make_dims(3), 1e-10, 1e-10, 1e-10);
        const auto hash2tables = make_indices(0, 1);
        const auto tablex      = make_tensor<scalar_t>(make_dims(2, 1, 1, 1), +1.42, -0.42);
        const auto hashes      = make_hashes(make_tensor<int32_t>(make_dims(2), 0, 2));

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
    {
        const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -3.42, +2.02, -0.42);
        const auto noise       = make_tensor<scalar_t>(make_dims(3), 1e-10, 1e-10, 1e-10);
        const auto hash2tables = make_indices(0, 1, 2);
        const auto tablex      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -3.42, +2.02, -0.42);
        const auto hashes      = make_hashes(make_tensor<int32_t>(make_dims(3), 0, 1, 2));

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
}

UTEST_CASE(fit_predict_sclass_ksplit)
{
    using fixture_t = fixture_datasource_t<ksplit_table_wlearner_t>;

    const auto maker  = make_ksplit_wlearner;
    const auto noise  = make_tensor<scalar_t>(make_dims(3), 1e-10, 1e-10, 1e-10);
    const auto hashes = make_hashes(make_tensor<int32_t>(make_dims(3), 0, 1, 2));
    {
        UTEST_NAMED_CASE("ksplit=1");
        const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -0.6, -0.6, -0.6);
        const auto hash2tables = make_indices(0, 0, 0);
        const auto tablex      = make_tensor<scalar_t>(make_dims(1, 1, 1, 1), -0.6);

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
    {
        UTEST_NAMED_CASE("ksplit=2");
        const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -0.5, +1.0, -0.5);
        const auto hash2tables = make_indices(0, 1, 0);
        const auto tablex      = make_tensor<scalar_t>(make_dims(2, 1, 1, 1), -0.5, +1.0);

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
    {
        UTEST_NAMED_CASE("ksplit=3");
        const auto tables      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -1, +2, -3);
        const auto hash2tables = make_indices(0, 1, 2);
        const auto tablex      = make_tensor<scalar_t>(make_dims(3, 1, 1, 1), -1, +2, -3);

        const auto datasource0 = make_datasource<fixture_t>(90, 1, tablex, tables, hashes, hash2tables, noise, maker);
        const auto datasourceX = make_random_datasource(make_features_all_continuous());

        check_wlearner(datasource0, datasourceX);
    }
}

UTEST_END_MODULE()
