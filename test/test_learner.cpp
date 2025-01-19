#include <fixture/configurable.h>
#include <fixture/dataset.h>
#include <fixture/datasource/hits.h>
#include <fixture/datasource/random.h>
#include <fixture/learner.h>
#include <fixture/loss.h>

using namespace nano;

namespace
{
auto make_predictions(const dataset_t& dataset, indices_cmap_t samples)
{
    return make_full_tensor<scalar_t>(cat_dims(samples.size(), dataset.target_dims()), samples.mean());
}

auto make_features()
{
    return features_t{
        feature_t{"mclass"}.mclass(strings_t{"m00", "m01", "m02"}),
        feature_t{"sclass"}.sclass(strings_t{"s00", "s01", "s02"}),
        feature_t{"scalar"}.scalar(feature_type::int16),
        feature_t{"struct"}.scalar(feature_type::uint8, make_dims(1, 2, 2)),
    };
}

auto make_datasource(const tensor_size_t samples, const size_t target)
{
    const auto features = make_features();
    const auto hits     = make_random_hits(samples, static_cast<tensor_size_t>(features.size()), target);

    auto datasource = random_datasource_t{samples, features, target, hits};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

class fixture_learner_t final : public learner_t
{
public:
    void do_predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const override
    {
        outputs = make_predictions(dataset, samples);
    }

    void fit(const dataset_t& dataset) { learner_t::fit_dataset(dataset); }
};
} // namespace

UTEST_BEGIN_MODULE(test_learner)

UTEST_CASE(fit_predict)
{
    const auto loss          = make_loss();
    const auto train_samples = arange(0, 80);
    const auto valid_samples = arange(80, 100);

    const auto datasource1 = make_datasource(100, 0U);
    const auto datasource2 = make_datasource(100, 1U);
    const auto datasource3 = make_datasource(100, 2U);

    const auto dataset1 = make_dataset(datasource1);
    const auto dataset2 = make_dataset(datasource2);
    const auto dataset3 = make_dataset(datasource3);

    {
        const auto learner = check_stream(fixture_learner_t{});

        check_predict_fails(learner, dataset1, train_samples);
        check_predict_fails(learner, dataset2, train_samples);
        check_predict_fails(learner, dataset3, train_samples);

        check_evaluate_fails(learner, dataset1, train_samples, *loss);
        check_evaluate_fails(learner, dataset2, train_samples, *loss);
        check_evaluate_fails(learner, dataset3, train_samples, *loss);
    }
    {
        const auto learner = check_stream(check_fit<fixture_learner_t>(dataset1));

        check_predict(learner, dataset1, train_samples, make_predictions(dataset1, train_samples));
        check_predict(learner, dataset1, valid_samples, make_predictions(dataset1, valid_samples));

        check_predict_fails(learner, dataset2, train_samples);
        check_predict_fails(learner, dataset3, train_samples);

        check_evaluate_fails(learner, dataset2, train_samples, *loss);
        check_evaluate_fails(learner, dataset3, train_samples, *loss);
    }
    {
        const auto learner = check_stream(check_fit<fixture_learner_t>(dataset2));

        check_predict(learner, dataset2, train_samples, make_predictions(dataset2, train_samples));
        check_predict(learner, dataset2, valid_samples, make_predictions(dataset2, valid_samples));

        check_predict_fails(learner, dataset1, train_samples);
        check_predict_fails(learner, dataset3, train_samples);

        check_evaluate_fails(learner, dataset1, train_samples, *loss);
        check_evaluate_fails(learner, dataset3, train_samples, *loss);
    }
    {
        const auto learner = check_stream(check_fit<fixture_learner_t>(dataset3));

        check_predict(learner, dataset3, train_samples, make_predictions(dataset3, train_samples));
        check_predict(learner, dataset3, valid_samples, make_predictions(dataset3, valid_samples));

        check_predict_fails(learner, dataset1, train_samples);
        check_predict_fails(learner, dataset2, train_samples);

        check_evaluate_fails(learner, dataset1, train_samples, *loss);
        check_evaluate_fails(learner, dataset2, train_samples, *loss);
    }
}

UTEST_END_MODULE()
