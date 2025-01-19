#include <fixture/wlearner.h>
#include <nano/wlearner/affine.h>
#include <nano/wlearner/dtree.h>
#include <nano/wlearner/table.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 1)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_feature() { return 5; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_weight() { return +1.42; }

    static auto expected_bias() { return -0.573; }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_weight(), expected_bias());
    }

    static auto make_wlearner() { return affine_wlearner_t{}; }

    static auto make_compatible_wlearners()
    {
        auto wlearners = rwlearners_t{};
        wlearners.emplace_back(make_wlearner().clone());
        return wlearners;
    }

    static auto make_incompatible_wlearners()
    {
        auto wlearners = rwlearners_t{};
        wlearners.emplace_back(dtree_wlearner_t{}.clone());
        wlearners.emplace_back(dense_table_wlearner_t{}.clone());
        return wlearners;
    }

    static void check_wlearner(const affine_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-13);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature = expected_feature();
        const auto fvalues = make_random_tensor<scalar_t>(make_dims(samples()), -1.0, +0.8);

        set_targets(feature, [&](const tensor_size_t sample)
                    { return make_affine_target(fvalues(sample), expected_weight(), expected_bias()); });
    }
};

UTEST_BEGIN_MODULE(test_wlearner_affine)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(100);
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner(datasource0, datasourceX);
}

UTEST_END_MODULE()
