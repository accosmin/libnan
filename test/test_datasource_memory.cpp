#include <fixture/datasource.h>
#include <nano/datasource.h>

using namespace nano;

namespace
{
auto make_features()
{
    return features_t{
        feature_t{"i8"}.scalar(feature_type::int8),
        feature_t{"i16"}.scalar(feature_type::int16),
        feature_t{"i32"}.scalar(feature_type::int32),
        feature_t{"i64"}.scalar(feature_type::int64),
        feature_t{"f32"}.scalar(feature_type::float32),
        feature_t{"f64"}.scalar(feature_type::float64),

        feature_t{"ui8_struct"}.scalar(feature_type::uint8, make_dims(2, 1, 2)),
        feature_t{"ui16_struct"}.scalar(feature_type::uint16, make_dims(1, 1, 1)),
        feature_t{"ui32_struct"}.scalar(feature_type::uint32, make_dims(1, 2, 1)),
        feature_t{"ui64_struct"}.scalar(feature_type::uint64, make_dims(1, 1, 2)),

        feature_t{"sclass2"}.sclass(2),
        feature_t{"sclass10"}.sclass(10),

        feature_t{"mclass3"}.mclass(3),
    };
}

class fixture_datasource_t final : public datasource_t
{
public:
    fixture_datasource_t(const tensor_size_t samples, features_t features, const size_t target)
        : datasource_t("fixture")
        , m_samples(samples)
        , m_features(std::move(features))
        , m_target(target)
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    void actually_do_load(bool do_load) { m_do_load = do_load; }

    static auto mask() { return make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask0() const { return m_target == 0U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask1() const { return m_target == 1U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0x80); }

    auto mask2() const { return m_target == 2U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x92, 0x49, 0x24, 0x80); }

    auto mask3() const { return m_target == 3U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x88, 0x88, 0x88, 0x80); }

    auto mask4() const { return m_target == 4U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x84, 0x21, 0x08, 0x00); }

    auto mask5() const { return m_target == 5U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x82, 0x08, 0x20, 0x80); }

    auto mask6() const { return m_target == 6U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask7() const { return m_target == 7U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask8() const { return m_target == 8U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask9() const { return m_target == 9U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0x80); }

    auto mask10() const
    {
        return m_target == 10U ? mask() : make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0x80);
    }

    auto mask11() const
    {
        return m_target == 11U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x92, 0x49, 0x24, 0x80);
    }

    auto mask12() const
    {
        return m_target == 12U ? mask() : make_tensor<uint8_t>(make_dims(4), 0x88, 0x88, 0x88, 0x80);
    }

    auto data0() const // NOLINT(readability-convert-member-functions-to-static)
    {
        return make_tensor<int8_t>(make_dims(25, 1, 1, 1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                   18, 19, 20, 21, 22, 23, 24);
    }

    auto data1() const
    {
        return m_target == 1U ? make_tensor<int16_t>(make_dims(25, 1, 1, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
                              : make_tensor<int16_t>(make_dims(25, 1, 1, 1), 1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 11, 0, 13, 0,
                                                     15, 0, 17, 0, 19, 0, 21, 0, 23, 0, 25);
    }

    auto data2() const
    {
        return m_target == 2U ? make_tensor<int32_t>(make_dims(25, 1, 1, 1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26)
                              : make_tensor<int32_t>(make_dims(25, 1, 1, 1), 2, 0, 0, 5, 0, 0, 8, 0, 0, 11, 0, 0, 14, 0,
                                                     0, 17, 0, 0, 20, 0, 0, 23, 0, 0, 26);
    }

    auto data3() const
    {
        return m_target == 3U ? make_tensor<int64_t>(make_dims(25, 1, 1, 1), 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
                              : make_tensor<int64_t>(make_dims(25, 1, 1, 1), 3, 0, 0, 0, 7, 0, 0, 0, 11, 0, 0, 0, 15, 0,
                                                     0, 0, 19, 0, 0, 0, 23, 0, 0, 0, 27);
    }

    auto data4() const
    {
        return m_target == 4U ? make_tensor<float>(make_dims(25, 1, 1, 1), 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28)
                              : make_tensor<float>(make_dims(25, 1, 1, 1), 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 14, 0, 0, 0, 0,
                                                   19, 0, 0, 0, 0, 24, 0, 0, 0, 0);
    }

    auto data5() const
    {
        return m_target == 5U ? make_tensor<double>(make_dims(25, 1, 1, 1), 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
                              : make_tensor<double>(make_dims(25, 1, 1, 1), 5, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 17, 0,
                                                    0, 0, 0, 0, 23, 0, 0, 0, 0, 0, 29);
    }

    auto data6() const // NOLINT(readability-convert-member-functions-to-static)
    {
        return make_tensor<uint8_t>(make_dims(25, 2, 1, 2), 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                    5, 5, 5, 5, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                                    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0,
                                    1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0, 0, 0, 0);
    }

    auto data7() const // NOLINT(readability-convert-member-functions-to-static)
    {
        return make_tensor<uint16_t>(make_dims(25, 1, 1, 1), 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5,
                                     6, 0, 1, 2, 3);
    }

    auto data8() const // NOLINT(readability-convert-member-functions-to-static)
    {
        return make_tensor<uint32_t>(make_dims(25, 1, 2, 1), 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1,
                                     2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
                                     0, 0);
    }

    auto data9() const // NOLINT(readability-convert-member-functions-to-static)
    {
        return make_tensor<uint64_t>(make_dims(25, 1, 1, 2), 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0,
                                     1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
                                     6, 6);
    }

    auto data10() const // NOLINT(readability-convert-member-functions-to-static)
    {
        return m_target == 10U ? make_tensor<uint8_t>(make_dims(25), 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                                      1, 0, 1, 0, 1, 0, 1, 0)
                               : make_tensor<uint8_t>(make_dims(25), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0);
    }

    auto data11() const
    {
        return m_target == 11U ? make_tensor<uint8_t>(make_dims(25), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6,
                                                      7, 8, 9, 0, 1, 2, 3, 4)
                               : make_tensor<uint8_t>(make_dims(25), 0, 0, 0, 3, 0, 0, 6, 0, 0, 9, 0, 0, 2, 0, 0, 5, 0,
                                                      0, 8, 0, 0, 1, 0, 0, 4);
    }

    auto data12() const
    {
        return m_target == 12U
                 ? make_tensor<uint8_t>(make_dims(25, 3), 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0,
                                        1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0,
                                        1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0)
                 : make_tensor<uint8_t>(make_dims(25, 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0);
    }

private:
    void do_load() override
    {
        resize(m_samples, m_features, m_target);

        if (!m_do_load)
        {
            return;
        }

        const auto itarget = static_cast<tensor_size_t>(m_target);

        // scalars
        for (tensor_size_t feature = 0; feature < 6; ++feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; sample += (itarget == feature) ? 1 : feature + 1)
            {
                this->set(sample, feature, sample + feature);
            }
        }

        // structured
        for (tensor_size_t feature = 6; feature < 10; ++feature)
        {
            for (tensor_size_t sample = 0; sample < m_samples; ++sample)
            {
                this->set(
                    sample, feature,
                    make_full_tensor<tensor_size_t>(m_features[static_cast<size_t>(feature)].dims(), sample % feature));
            }
        }

        // single label
        for (tensor_size_t sample = 0, feature = 10; sample < m_samples; sample += (itarget == feature) ? 1 : 2)
        {
            this->set(sample, feature, sample % 2);
        }
        for (tensor_size_t sample = 0, feature = 11; sample < m_samples; sample += (itarget == feature) ? 1 : 3)
        {
            this->set(sample, feature, sample % 10);
        }

        // multi label
        for (tensor_size_t sample = 0, feature = 12; sample < m_samples; sample += (itarget == feature) ? 1 : 4)
        {
            this->set(sample, feature, make_full_tensor<uint8_t>(make_dims(3), sample % 3));
        }
    }

    tensor_size_t m_samples{0};
    features_t    m_features;
    size_t        m_target;
    bool          m_do_load{true};
};

auto make_datasource(tensor_size_t samples, const features_t& features, size_t target)
{
    auto datasource = fixture_datasource_t{samples, features, target};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

template <class tdata>
void check_inputs_or_target(const datasource_t& datasource, const features_t& features, const size_t ifeature,
                            const size_t target, const tdata& data, const mask_cmap_t& mask)
{
    if (ifeature == target)
    {
        check_target(datasource, features[ifeature], data, mask);
    }
    else
    {
        check_inputs(datasource, static_cast<tensor_size_t>(ifeature < target ? ifeature : (ifeature - 1U)),
                     features[ifeature], data, mask);
    }
}
} // namespace

UTEST_BEGIN_MODULE(test_datasource_memory)

UTEST_CASE(check_samples)
{
    const auto features   = make_features();
    const auto samples    = ::nano::arange(0, 100);
    auto       datasource = make_datasource(samples.size(), features, string_t::npos);
    {
        const auto test_samples = datasource.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = datasource.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
    {
        datasource.testing(make_range(0, 10));
        datasource.testing(make_range(20, 50));

        const auto test_samples = datasource.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 40);
        UTEST_CHECK_EQUAL(test_samples.slice(0, 10), ::nano::arange(0, 10));
        UTEST_CHECK_EQUAL(test_samples.slice(10, 40), ::nano::arange(20, 50));

        const auto train_samples = datasource.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 60);
        UTEST_CHECK(train_samples.slice(0, 10) == ::nano::arange(10, 20));
        UTEST_CHECK(train_samples.slice(10, 60) == ::nano::arange(50, 100));
    }
    {
        datasource.no_testing();

        const auto test_samples = datasource.test_samples();
        UTEST_CHECK_EQUAL(test_samples.size(), 0);

        const auto train_samples = datasource.train_samples();
        UTEST_CHECK_EQUAL(train_samples.size(), 100);
        UTEST_CHECK_EQUAL(train_samples, ::nano::arange(0, 100));
    }
}

UTEST_CASE(datasource_target_NA)
{
    const auto features   = make_features();
    const auto samples    = ::nano::arange(0, 25);
    const auto datasource = make_datasource(samples.size(), features, string_t::npos);

    UTEST_CHECK_EQUAL(datasource.features(), 13);
    UTEST_CHECK_EQUAL(datasource.type(), task_type::unsupervised);

    check_inputs(datasource, 0, features[0U], datasource.data0(), datasource.mask0());
    check_inputs(datasource, 1, features[1U], datasource.data1(), datasource.mask1());
    check_inputs(datasource, 2, features[2U], datasource.data2(), datasource.mask2());
    check_inputs(datasource, 3, features[3U], datasource.data3(), datasource.mask3());
    check_inputs(datasource, 4, features[4U], datasource.data4(), datasource.mask4());
    check_inputs(datasource, 5, features[5U], datasource.data5(), datasource.mask5());
    check_inputs(datasource, 6, features[6U], datasource.data6(), datasource.mask6());
    check_inputs(datasource, 7, features[7U], datasource.data7(), datasource.mask7());
    check_inputs(datasource, 8, features[8U], datasource.data8(), datasource.mask8());
    check_inputs(datasource, 9, features[9U], datasource.data9(), datasource.mask9());
    check_inputs(datasource, 10, features[10U], datasource.data10(), datasource.mask10());
    check_inputs(datasource, 11, features[11U], datasource.data11(), datasource.mask11());
    check_inputs(datasource, 12, features[12U], datasource.data12(), datasource.mask12());
}

UTEST_CASE(datasource_target)
{
    const auto features = make_features();
    const auto samples  = ::nano::arange(0, 25);
    for (size_t target = 0U; target < 13U; ++target)
    {
        const auto datasource = make_datasource(samples.size(), features, target);

        UTEST_CHECK_EQUAL(datasource.features(), 12);
        UTEST_CHECK_EQUAL(datasource.type(), target < 10U   ? task_type::regression
                                             : target < 12U ? task_type::sclassification
                                                            : task_type::mclassification);

        check_inputs_or_target(datasource, features, 0U, target, datasource.data0(), datasource.mask0());
        check_inputs_or_target(datasource, features, 1U, target, datasource.data1(), datasource.mask1());
        check_inputs_or_target(datasource, features, 2U, target, datasource.data2(), datasource.mask2());
        check_inputs_or_target(datasource, features, 3U, target, datasource.data3(), datasource.mask3());
        check_inputs_or_target(datasource, features, 4U, target, datasource.data4(), datasource.mask4());
        check_inputs_or_target(datasource, features, 5U, target, datasource.data5(), datasource.mask5());
        check_inputs_or_target(datasource, features, 6U, target, datasource.data6(), datasource.mask6());
        check_inputs_or_target(datasource, features, 7U, target, datasource.data7(), datasource.mask7());
        check_inputs_or_target(datasource, features, 8U, target, datasource.data8(), datasource.mask8());
        check_inputs_or_target(datasource, features, 9U, target, datasource.data9(), datasource.mask9());
        check_inputs_or_target(datasource, features, 10U, target, datasource.data10(), datasource.mask10());
        check_inputs_or_target(datasource, features, 11U, target, datasource.data11(), datasource.mask11());
        check_inputs_or_target(datasource, features, 12U, target, datasource.data12(), datasource.mask12());
    }
}

UTEST_CASE(invalid_feature_type)
{
    auto features = make_features();
    features[0].scalar(static_cast<feature_type>(0xFF));

    auto datasource = fixture_datasource_t{100, features, string_t::npos};
    UTEST_CHECK_THROW(datasource.load(), std::runtime_error);

    datasource.actually_do_load(false);
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_THROW(datasource.visit_inputs(0, [](const auto&, const auto&, const auto&) {}), std::runtime_error);
}

UTEST_CASE(invalid_targets_type)
{
    auto features = make_features();
    features[0].scalar(static_cast<feature_type>(0xFF));

    auto datasource = fixture_datasource_t{100, features, string_t::npos};
    UTEST_CHECK_THROW(datasource.load(), std::runtime_error);

    datasource.actually_do_load(false);
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_THROW(datasource.visit_inputs(0, [](const auto&, const auto&, const auto&) {}), std::runtime_error);
}

UTEST_END_MODULE()
