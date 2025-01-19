#include <fixture/wlearner.h>
#include <nano/wlearner/affine.h>
#include <nano/wlearner/stump.h>
#include <nano/wlearner/table.h>

using namespace nano;

class fixture_datasource_t final : public wlearner_datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples)
        : wlearner_datasource_t(samples, 2)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    static auto expected_feature() { return 6; }

    static auto expected_features() { return make_indices(expected_feature()); }

    static auto expected_threshold() { return 2.5; }

    static auto expected_pred_lower() { return +3.0; }

    static auto expected_pred_upper() { return -2.1; }

    static auto expected_tables()
    {
        return make_tensor<scalar_t>(make_dims(2, 1, 1, 1), expected_pred_lower(), expected_pred_upper());
    }

    static auto make_wlearner() { return stump_wlearner_t{}; }

    static auto make_compatible_wlearners()
    {
        auto wlearners = rwlearners_t{};
        return wlearners;
    }

    static auto make_incompatible_wlearners()
    {
        auto wlearners = rwlearners_t{};
        wlearners.emplace_back(affine_wlearner_t{}.clone());
        wlearners.emplace_back(dense_table_wlearner_t{}.clone());
        wlearners.emplace_back(make_wlearner().clone());
        return wlearners;
    }

    static void check_wlearner(const stump_wlearner_t& wlearner)
    {
        UTEST_CHECK_EQUAL(wlearner.feature(), expected_feature());
        UTEST_CHECK_EQUAL(wlearner.features(), expected_features());
        UTEST_CHECK_CLOSE(wlearner.tables(), expected_tables(), 1e-13);
        UTEST_CHECK_CLOSE(wlearner.threshold(), expected_threshold(), 1e-13);
    }

private:
    void do_load() override
    {
        random_datasource_t::do_load();

        const auto feature = expected_feature();
        const auto fvalues = make_random_tensor<int32_t>(make_dims(this->samples()), -5, +4);

        set_targets(feature,
                    [&](const tensor_size_t sample) {
                        return make_stump_target(fvalues(sample), expected_threshold(), expected_pred_lower(),
                                                 expected_pred_upper());
                    });
    }
};

UTEST_BEGIN_MODULE(test_wlearner_stump)

UTEST_CASE(fit_predict)
{
    const auto datasource0 = make_datasource<fixture_datasource_t>(200);
    const auto datasourceX = make_random_datasource(make_features_all_discrete());

    check_wlearner(datasource0, datasourceX);
}

UTEST_END_MODULE()
