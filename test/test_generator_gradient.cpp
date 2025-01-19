#include <fixture/generator.h>
#include <nano/generator/elemwise_gradient.h>

using namespace std;
using namespace nano;

namespace
{
template <class tscalar>
auto make_input_data()
{
    return make_tensor<tscalar>(make_dims(2, 4, 4), 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 4, 4, 5, 0, 1, 1, 1, 0, 0, 1,
                                1, 0, 0, 1, 1, 1, 0, 0, 0);
}

// clang-format off

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define NaN4 NaN, NaN, NaN, NaN

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GX0(scale) (scale) * 2.00, (scale) * 2.00, (scale) * 1.50, (scale) * 1.75
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GX1(scale) (scale) * 1.00, (scale) * 0.75, (scale) * 0.50, (scale) * 0.75

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GY0(scale) (scale) * +2.00, (scale) * +2.00, (scale) * 1.00, (scale) * +0.25
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GY1(scale) (scale) * -0.50, (scale) * -0.25, (scale) * 0.00, (scale) * -0.75

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GG0(scale) (scale) * sqrt(8.00), (scale) * sqrt(8.000), (scale) * sqrt(3.25), (scale) * sqrt(3.125)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GG1(scale) (scale) * sqrt(1.25), (scale) * sqrt(0.625), (scale) * sqrt(0.25), (scale) * sqrt(1.125)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THETA0 atan2(+2.0, 2.0), atan2(+2.00, 2.00), atan2(1.0, 1.5), atan2(+0.25, 1.75)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define THETA1 atan2(-0.5, 1.0), atan2(-0.25, 0.75), atan2(0.0, 0.5), atan2(-0.75, 0.75)

// clang-format on

auto make_features(tensor_size_t channels = 2, tensor_size_t rows = 4, tensor_size_t cols = 4)
{
    return features_t{
        feature_t{"mclass3"}.mclass(strings_t{"m0", "m1", "m2"}),
        feature_t{"sclass2"}.sclass(strings_t{"s0", "s1"}),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"u8s"}.scalar(feature_type::uint8, make_dims(channels, rows, cols)),
        feature_t{"f64"}.scalar(feature_type::float64),
    };
}

class fixture_datasource_t final : public datasource_t
{
public:
    fixture_datasource_t(tensor_size_t samples, size_t target, tensor_size_t channels = 2, tensor_size_t rows = 4,
                         tensor_size_t cols = 4)
        : datasource_t("fixture")
        , m_samples(samples)
        , m_features(make_features(channels, rows, cols))
        , m_target(target)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

private:
    void do_load() override
    {
        resize(m_samples, m_features, m_target);

        for (tensor_size_t sample = 0; sample < m_samples; sample += 2)
        {
            if (m_features[3U].dims() == make_dims(2, 4, 4))
            {
                auto values = make_input_data<uint8_t>();
                values.array() *= static_cast<uint8_t>(sample + 1);
                set(sample, 3, values);
            }
        }
    }

    tensor_size_t m_samples{0};
    features_t    m_features;
    size_t        m_target;
};

auto make_datasource(tensor_size_t samples, size_t target, tensor_size_t channels = 2, tensor_size_t rows = 4,
                     tensor_size_t cols = 4)
{
    auto datasource = fixture_datasource_t{samples, target, channels, rows, cols};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

auto make_dataset(const datasource_t& datasource)
{
    auto dataset = dataset_t{datasource};
    add_generator<gradient_generator_t>(dataset);
    return dataset;
}
} // namespace

UTEST_BEGIN_MODULE(test_generator_gradient)

UTEST_CASE(kernel)
{
    {
        const auto kernel = make_kernel3x3<double>(kernel3x3_type::sobel);
        UTEST_CHECK_CLOSE(kernel[0], 1.0 / 4.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[1], 2.0 / 4.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[2], 1.0 / 4.0, 1e-15);
    }
    {
        const auto kernel = make_kernel3x3<double>(kernel3x3_type::scharr);
        UTEST_CHECK_CLOSE(kernel[0], 3.0 / 16.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[1], 10.0 / 16.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[2], 3.0 / 16.0, 1e-15);
    }
    {
        const auto kernel = make_kernel3x3<double>(kernel3x3_type::prewitt);
        UTEST_CHECK_CLOSE(kernel[0], 1.0 / 3.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[1], 1.0 / 3.0, 1e-15);
        UTEST_CHECK_CLOSE(kernel[2], 1.0 / 3.0, 1e-15);
    }
    {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        const auto kernel = make_kernel3x3<double>(static_cast<kernel3x3_type>(0xFF));
        UTEST_CHECK(!std::isfinite(kernel[0]));
        UTEST_CHECK(!std::isfinite(kernel[1]));
        UTEST_CHECK(!std::isfinite(kernel[2]));
    }
}

UTEST_CASE(gradient)
{
    const auto input = make_input_data<int>();

    const std::array<scalar_t, 3> kernel = {+0.25, +0.50, +0.25};

    auto output = tensor_mem_t<scalar_t, 2>(2, 2);
    {
        gradient3x3(gradient3x3_mode::gradx, input.tensor(0), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GX0(1));
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::gradx, input.tensor(1), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GX1(1));
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::grady, input.tensor(0), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GY0(1));
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::grady, input.tensor(1), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GY1(1));
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::magnitude, input.tensor(0), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GG0(1));
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::magnitude, input.tensor(1), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), GG1(1));
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::angle, input.tensor(0), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), THETA0);
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
    {
        gradient3x3(gradient3x3_mode::angle, input.tensor(1), kernel, output.tensor());
        const auto expected_output = make_tensor<scalar_t>(make_dims(2, 2), THETA1);
        UTEST_CHECK_CLOSE(output, expected_output, 1e-15);
    }
}

UTEST_CASE(unsupervised_gradient)
{
    const auto datasource = make_datasource(4, string_t::npos);
    const auto dataset    = make_dataset(datasource);

    UTEST_REQUIRE_EQUAL(dataset.features(), 8);
    UTEST_CHECK_EQUAL(dataset.feature(0),
                      feature_t{"sobel::gx(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(1),
                      feature_t{"sobel::gy(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(2),
                      feature_t{"sobel::gg(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(3),
                      feature_t{"sobel::theta(u8s[channel::0])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(4),
                      feature_t{"sobel::gx(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(5),
                      feature_t{"sobel::gy(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(6),
                      feature_t{"sobel::gg(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));
    UTEST_CHECK_EQUAL(dataset.feature(7),
                      feature_t{"sobel::theta(u8s[channel::1])"}.scalar(feature_type::float64, make_dims(1, 2, 2)));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), GX0(1), NaN4, GX0(3), NaN4));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), GY0(1), NaN4, GY0(3), NaN4));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), GG0(1), NaN4, GG0(3), NaN4));
    check_select(dataset, 3, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), THETA0, NaN4, THETA0, NaN4));
    check_select(dataset, 4, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), GX1(1), NaN4, GX1(3), NaN4));
    check_select(dataset, 5, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), GY1(1), NaN4, GY1(3), NaN4));
    check_select(dataset, 6, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), GG1(1), NaN4, GG1(3), NaN4));
    check_select(dataset, 7, make_tensor<scalar_t>(make_dims(4, 1, 2, 2), THETA1, NaN4, THETA1, NaN4));
    check_select_stats(dataset, indices_t{}, indices_t{}, indices_t{}, make_indices(0, 1, 2, 3, 4, 5, 6, 7));

    check_flatten(
        dataset,
        make_tensor<scalar_t>(make_dims(4, 32), GX0(1), GY0(1), GG0(1), THETA0, GX1(1), GY1(1), GG1(1), THETA1, NaN4,
                              NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, GX0(3), GY0(3), GG0(3), THETA0, GX1(3), GY1(3),
                              GG1(3), THETA1, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4),
        make_indices(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7));

    dataset.drop(1);
    check_flatten(
        dataset,
        make_tensor<scalar_t>(make_dims(4, 32), GX0(1), NaN4, GG0(1), THETA0, GX1(1), GY1(1), GG1(1), THETA1, NaN4,
                              NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, GX0(3), NaN4, GG0(3), THETA0, GX1(3), GY1(3),
                              GG1(3), THETA1, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4, NaN4),
        make_indices(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7),
        true);
}

UTEST_CASE(unsupervised_too_small_rows)
{
    const auto datasource = make_datasource(4, string_t::npos, 2, 2, 4);
    const auto dataset    = make_dataset(datasource);

    UTEST_CHECK_EQUAL(dataset.features(), 0);
}

UTEST_CASE(unsupervised_too_small_cols)
{
    const auto datasource = make_datasource(4, string_t::npos, 2, 4, 2);
    const auto dataset    = make_dataset(datasource);

    UTEST_CHECK_EQUAL(dataset.features(), 0);
}

UTEST_CASE(unsupervised_too_small_rows_and_cols)
{
    const auto datasource = make_datasource(4, string_t::npos, 2, 2, 2);
    const auto dataset    = make_dataset(datasource);

    UTEST_CHECK_EQUAL(dataset.features(), 0);
}

UTEST_END_MODULE()
