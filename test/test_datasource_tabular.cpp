#include <filesystem>
#include <fixture/datasource.h>
#include <fstream>
#include <nano/datasource/tabular.h>

using namespace nano;

namespace
{
auto feature_cont()
{
    return feature_t{"cont"}.scalar(feature_type::float64);
}

auto feature_cate(bool with_labels = false)
{
    auto feature = feature_t{"cate"};
    if (with_labels)
    {
        feature.sclass(strings_t{"cate0", "cate1", "cate2"});
    }
    else
    {
        feature.sclass(3);
    }
    return feature;
}

class fixture_datasource_t final : public tabular_datasource_t
{
public:
    static auto data_path() { return "test_datasource_tabular_data.csv"; }

    static auto test_path() { return "test_datasource_tabular_test.csv"; }

    static auto csvs(int data_size = 20, int test_size = 10)
    {
        return csvs_t({csv_t{data_path()}.delim(",").header(false).expected(data_size).skip('@').placeholder("?"),
                       csv_t{test_path()}
                           .delim(",")
                           .header(true)
                           .expected(test_size)
                           .skip('@')
                           .testing(0, test_size)
                           .placeholder("?")});
    }

    fixture_datasource_t()
        : fixture_datasource_t(fixture_datasource_t::csvs(), features_t{})
    {
    }

    explicit fixture_datasource_t(features_t features)
        : fixture_datasource_t(fixture_datasource_t::csvs(), std::move(features))
    {
    }

    fixture_datasource_t(features_t features, size_t target)
        : fixture_datasource_t(fixture_datasource_t::csvs(), std::move(features), target)
    {
    }

    fixture_datasource_t(csvs_t csvs, features_t features)
        : tabular_datasource_t("fixture", std::move(csvs), std::move(features))
    {
        std::filesystem::remove(data_path()); // NOLINT(cert-err33-c)
        std::filesystem::remove(test_path()); // NOLINT(cert-err33-c)
    }

    fixture_datasource_t(csvs_t csvs, features_t features, size_t target)
        : tabular_datasource_t("fixture", std::move(csvs), std::move(features), target)
        , m_target(target)
    {
        std::filesystem::remove(data_path()); // NOLINT(cert-err33-c)
        std::filesystem::remove(test_path()); // NOLINT(cert-err33-c)
    }

    fixture_datasource_t(fixture_datasource_t&&)                 = default;
    fixture_datasource_t(const fixture_datasource_t&)            = default;
    fixture_datasource_t& operator=(fixture_datasource_t&&)      = default;
    fixture_datasource_t& operator=(const fixture_datasource_t&) = delete;

    ~fixture_datasource_t() override
    {
        std::filesystem::remove(data_path()); // NOLINT(cert-err33-c)
        std::filesystem::remove(test_path()); // NOLINT(cert-err33-c)
    }

    void too_many_labels() { m_too_many_labels = true; }

    void optional_target() { m_optional_target = true; }

    void mandatory_target() { m_optional_target = false; }

    void prepare()
    {
        std::cout << "target=" << m_target << ", optional_target=" << m_optional_target
                  << ", optional_cont=" << optional_cont() << ", optional_cate=" << optional_cate() << "\n";
        write_data(data_path());
        write_test(test_path());
    }

    auto mask_cate() const
    {
        return optional_cate() ? make_tensor<uint8_t>(make_dims(4), 0xEF, 0x7B, 0xDE, 0xF4)
                               : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0xFC);
    }

    auto mask_cont() const
    {
        return optional_cont() ? make_tensor<uint8_t>(make_dims(4), 0xAA, 0xAA, 0xAA, 0xA8)
                               : make_tensor<uint8_t>(make_dims(4), 0xFF, 0xFF, 0xFF, 0xFC);
    }

    auto values_cate() const
    {
        return optional_cate() ? make_tensor<uint8_t>(make_dims(30), 0, 1, 2, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0, 1,
                                                      2, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 0, 2)
                               : make_tensor<uint8_t>(make_dims(30), 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
                                                      2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2);
    }

    auto values_cont() const
    {
        return optional_cont()
                 ? make_tensor<scalar_t>(make_dims(30, 1, 1, 1), +2.8, +0.0, +2.4, +0.0, +2.0, +0.0, +1.6, +0.0, +1.2,
                                         +0.0, +0.8, +0.0, +0.4, +0.0, +0.0, +0.0, -0.4, +0.0, -0.8, +0.0, -1.2, +0.0,
                                         -1.6, +0.0, -2.0, +0.0, -2.4, +0.0, -2.8, +0.0)
                 : make_tensor<scalar_t>(make_dims(30, 1, 1, 1), +2.8, +2.6, +2.4, +2.2, +2.0, +1.8, +1.6, +1.4, +1.2,
                                         +1.0, +0.8, +0.6, +0.4, +0.2, +0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4,
                                         -1.6, -1.8, -2.0, -2.2, -2.4, -2.6, -2.8, -3.0);
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

private:
    bool optional_cont() const { return m_optional_target || m_target == 1U; }

    bool optional_cate() const { return m_optional_target || m_target == 0U; }

    void write_data(const char* path) const
    {
        std::ofstream os(path);
        write_tab(os, 1, 20, false);
        UTEST_REQUIRE(os);
    }

    void write_test(const char* path) const
    {
        std::ofstream os(path);
        write_tab(os, 21, 10, true);
        UTEST_REQUIRE(os);
    }

    void write_tab(std::ostream& stream, const int begin, const int size, const bool header) const
    {
        if (header)
        {
            stream << "cont,cate\n";
        }

        for (auto index = begin; index < begin + size; ++index)
        {
            (index % 2 == 0 && optional_cont()) ? (stream << "?,") : (stream << (3.0 - 0.2 * index) << ",");
            (index % 5 == 4 && optional_cate())
                ? (stream << "?,")
                : (stream << "cate" << ((index - 1) % (m_too_many_labels ? 4 : 3)) << ",");
            stream << "\n";

            if (index % 7 == 0)
            {
                stream << "\n";
            }
            if (index % 9 == 0)
            {
                stream << "@ this line should be skipped\n";
            }
        }
    }

    // attributes
    size_t m_target{string_t::npos}; ///<
    bool   m_too_many_labels{false}; ///< toggle whether to write an invalid number of labels for categorical features
    bool   m_optional_target{true};  ///< optional
};
} // namespace

UTEST_BEGIN_MODULE(test_datasource_tabular)

UTEST_CASE(empty)
{
    auto dataset = fixture_datasource_t{};
    UTEST_REQUIRE_NOTHROW(dataset.prepare());

    UTEST_CHECK_EQUAL(dataset.samples(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.test_samples(), indices_t{});
    UTEST_CHECK_EQUAL(dataset.train_samples(), indices_t{});
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);
}

UTEST_CASE(no_target_no_load)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()}
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());

    UTEST_CHECK_EQUAL(dataset.samples(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);
    UTEST_CHECK_EQUAL(dataset.test_samples(), indices_t{});
    UTEST_CHECK_EQUAL(dataset.train_samples(), indices_t{});
}

UTEST_CASE(with_target_no_load)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()},
        0U
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());

    UTEST_CHECK_EQUAL(dataset.samples(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);
    UTEST_CHECK_EQUAL(dataset.test_samples(), indices_t{});
    UTEST_CHECK_EQUAL(dataset.train_samples(), indices_t{});
}

UTEST_CASE(cannot_load_no_data)
{
    const auto csvs    = csvs_t{};
    auto       dataset = fixture_datasource_t{
        csvs, {feature_cont(), feature_cate()},
         0U
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_no_features)
{
    auto dataset = fixture_datasource_t{};
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_invalid_target)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()},
        2U
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_unsupported_mclass)
{
    const auto feature_mclass = feature_t{"feature"}.mclass(3);

    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate(), feature_mclass}
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_unsupported_struct)
{
    const auto feature_struct = feature_t{"feature"}.scalar(feature_type::uint8, make_dims(3, 32, 32));

    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate(), feature_struct}
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_wrong_expected_csv_length0)
{
    const auto csvs    = fixture_datasource_t::csvs(21, 10);
    auto       dataset = fixture_datasource_t{
        csvs, {feature_cont(), feature_cate()}
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_wrong_expected_csv_length1)
{
    const auto csvs    = fixture_datasource_t::csvs(20, 9);
    auto       dataset = fixture_datasource_t{
        csvs, {feature_cont(), feature_cate()}
    };
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(cannot_load_too_many_labels)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()},
        string_t::npos
    };
    dataset.too_many_labels();
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(load_no_target)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()}
    };
    dataset.parameter("datasource::basedir") = "";
    dataset.optional_target();
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.features(), 2);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_cate(true));
    UTEST_CHECK_EQUAL(dataset.test_samples(), arange(20, 30));
    UTEST_CHECK_EQUAL(dataset.train_samples(), arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.type(), task_type::unsupervised);

    check_inputs(dataset, 0, feature_cont(), dataset.values_cont(), dataset.mask_cont());
    check_inputs(dataset, 1, feature_cate(true), dataset.values_cate(), dataset.mask_cate());
}

UTEST_CASE(load_cate_target)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()},
        1U
    };
    dataset.parameter("datasource::basedir") = ".";
    dataset.mandatory_target();
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.features(), 1);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cont());
    UTEST_CHECK_EQUAL(dataset.test_samples(), arange(20, 30));
    UTEST_CHECK_EQUAL(dataset.train_samples(), arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.type(), task_type::sclassification);

    check_inputs(dataset, 0, feature_cont(), dataset.values_cont(), dataset.mask_cont());
    check_target(dataset, feature_cate(true), dataset.values_cate(), dataset.mask_cate());

    dataset.optional_target();
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_CHECK_THROW(dataset.load(), std::runtime_error);
}

UTEST_CASE(load_cont_target)
{
    auto dataset = fixture_datasource_t{
        {feature_cont(), feature_cate()},
        0U
    };
    dataset.parameter("datasource::basedir") = ".";
    dataset.mandatory_target();
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_REQUIRE_NOTHROW(dataset.load());
    UTEST_CHECK_EQUAL(dataset.samples(), 30);
    UTEST_CHECK_EQUAL(dataset.features(), 1);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_cate(true));
    UTEST_CHECK_EQUAL(dataset.test_samples(), arange(20, 30));
    UTEST_CHECK_EQUAL(dataset.train_samples(), arange(0, 20));
    UTEST_CHECK_EQUAL(dataset.type(), task_type::regression);

    check_target(dataset, feature_cont(), dataset.values_cont(), dataset.mask_cont());
    check_inputs(dataset, 0, feature_cate(true), dataset.values_cate(), dataset.mask_cate());

    dataset.optional_target();
    UTEST_REQUIRE_NOTHROW(dataset.prepare());
    UTEST_CHECK_THROW(dataset.load(), std::runtime_error);
}

UTEST_END_MODULE()
