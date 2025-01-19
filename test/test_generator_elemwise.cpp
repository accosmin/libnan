#include <fixture/generator.h>
#include <fixture/generator_datasource.h>
#include <nano/generator/elemwise.h>

using namespace nano;

class NANO_PUBLIC scalar_to_scalar_t : public elemwise_input_scalar_t, public generated_scalar_t
{
public:
    explicit scalar_to_scalar_t(indices_t features = indices_t{})
        : elemwise_input_scalar_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "feature"); }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values) { return static_cast<scalar_t>(values(0)) < 0.0 ? -1.0 : +1.0; };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC scalar_to_sclass_t : public elemwise_input_scalar_t, public generated_sclass_t
{
public:
    explicit scalar_to_sclass_t(indices_t features = indices_t{})
        : elemwise_input_scalar_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_sclass_feature(ifeature, "feature", strings_t{"neg", "pos"});
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values) { return static_cast<scalar_t>(values(0)) < 0.0 ? 0 : 1; };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC scalar_to_mclass_t : public elemwise_input_scalar_t, public generated_mclass_t
{
public:
    explicit scalar_to_mclass_t(indices_t features = indices_t{})
        : elemwise_input_scalar_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_mclass_feature(ifeature, "feature", strings_t{"odd", "even", "div3"});
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{3};
        const auto process = [=](const auto& values, auto&& mclass)
        {
            mclass(0) = (static_cast<int>(values(0)) % 2) != 0 ? 0x01 : 0x00;
            mclass(1) = (static_cast<int>(values(0)) % 2) == 0 ? 0x01 : 0x00;
            mclass(2) = (static_cast<int>(values(0)) % 3) == 0 ? 0x01 : 0x00;
        };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC scalar_to_struct_t : public elemwise_input_scalar_t, public generated_struct_t
{
public:
    explicit scalar_to_struct_t(indices_t features = indices_t{})
        : elemwise_input_scalar_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_struct_feature(ifeature, "feature", make_dims(4, 1, 1));
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{4};
        const auto process = [=](const auto& values, auto&& structured)
        {
            const auto svalue = static_cast<scalar_t>(values(0));
            structured(0)     = svalue;
            structured(1)     = svalue * svalue;
            structured(2)     = svalue * svalue * svalue;
            structured(3)     = svalue * svalue * svalue * svalue;
        };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC struct_to_scalar_t : public elemwise_input_struct_t, public generated_scalar_t
{
public:
    explicit struct_to_scalar_t(indices_t features = indices_t{})
        : elemwise_input_struct_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "feature"); }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values) { return values.array().template cast<scalar_t>().sum(); };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC struct_to_sclass_t : public elemwise_input_struct_t, public generated_sclass_t
{
public:
    explicit struct_to_sclass_t(indices_t features = indices_t{})
        : elemwise_input_struct_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_sclass_feature(ifeature, "feature", strings_t{"<10", ">=10"});
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values)
        { return values.array().template cast<scalar_t>().sum() < 10.0 ? 0 : 1; };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC struct_to_mclass_t : public elemwise_input_struct_t, public generated_mclass_t
{
public:
    explicit struct_to_mclass_t(indices_t features = indices_t{})
        : elemwise_input_struct_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_mclass_feature(ifeature, "feature", strings_t{"<10", "<30", "<20"});
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{3};
        const auto process = [=](const auto& values, auto&& mclass)
        {
            const auto sum = values.array().template cast<scalar_t>().sum();
            mclass(0)      = (sum < 10.0) ? 0x01 : 0x00;
            mclass(1)      = (sum < 30.0) ? 0x01 : 0x00;
            mclass(2)      = (sum < 20.0) ? 0x01 : 0x00;
        };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC struct_to_struct_t : public elemwise_input_struct_t, public generated_struct_t
{
public:
    explicit struct_to_struct_t(indices_t features = indices_t{})
        : elemwise_input_struct_t("gg", std::move(features))
    {
    }

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_struct_feature(ifeature, "feature", make_dims(2, 1, 1));
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{2};
        const auto process = [=](const auto& values, auto&& structured)
        {
            const auto sum = values.array().template cast<scalar_t>().sum();
            structured(0)  = sum;
            structured(1)  = sum + 1;
        };
        return std::make_tuple(process, colsize);
    }
};

UTEST_BEGIN_MODULE(test_generator_elemwise)

UTEST_CASE(scalar)
{
    const auto datasource = make_datasource(10, string_t::npos);

    auto dataset = dataset_t{datasource};
    add_generator<elemwise_generator_t<scalar_to_scalar_t>>(dataset);
    add_generator<elemwise_generator_t<scalar_to_sclass_t>>(dataset);
    add_generator<elemwise_generator_t<scalar_to_mclass_t>>(dataset);
    add_generator<elemwise_generator_t<scalar_to_struct_t>>(dataset);

    UTEST_REQUIRE_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"feature(scalar0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"feature(scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"feature(scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_t{"feature(scalar0)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(4), feature_t{"feature(scalar1)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(5), feature_t{"feature(scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(6), feature_t{"feature(scalar0)"}.mclass(strings_t{"odd", "even", "div3"}));
    UTEST_CHECK_EQUAL(dataset.feature(7), feature_t{"feature(scalar1)"}.mclass(strings_t{"odd", "even", "div3"}));
    UTEST_CHECK_EQUAL(dataset.feature(8), feature_t{"feature(scalar2)"}.mclass(strings_t{"odd", "even", "div3"}));
    UTEST_CHECK_EQUAL(dataset.feature(9),
                      feature_t{"feature(scalar0)"}.scalar(feature_type::float64, make_dims(4, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(10),
                      feature_t{"feature(scalar1)"}.scalar(feature_type::float64, make_dims(4, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(11),
                      feature_t{"feature(scalar2)"}.scalar(feature_type::float64, make_dims(4, 1, 1)));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(10), -1, +1, +1, +1, +1, +1, +1, +1, +1, +1));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(10), -1, Na, +1, Na, +1, Na, +1, Na, +1, Na));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(10), -1, Na, Na, +1, Na, Na, +1, Na, Na, +1));
    check_select(dataset, 3, make_tensor<int32_t>(make_dims(10), +0, +1, +1, +1, +1, +1, +1, +1, +1, +1));
    check_select(dataset, 4, make_tensor<int32_t>(make_dims(10), +0, -1, +1, -1, +1, -1, +1, -1, +1, -1));
    check_select(dataset, 5, make_tensor<int32_t>(make_dims(10), +0, -1, -1, +1, -1, -1, +1, -1, -1, +1));
    check_select(dataset, 6,
                 make_tensor<int8_t>(make_dims(10, 3), +1, +0, +0, +0, +1, +1, +1, +0, +0, +0, +1, +0, +1, +0, +1, +0,
                                     +1, +0, +1, +0, +0, +0, +1, +1, +1, +0, +0, +0, +1, +0));
    check_select(dataset, 7,
                 make_tensor<int8_t>(make_dims(10, 3), +0, +1, +0, -1, -1, -1, +0, +1, +1, -1, -1, -1, +0, +1, +0, -1,
                                     -1, -1, +0, +1, +0, -1, -1, -1, +0, +1, +1, -1, -1, -1));
    check_select(dataset, 8,
                 make_tensor<int8_t>(make_dims(10, 3), +1, +0, +1, -1, -1, -1, -1, -1, -1, +0, +1, +1, -1, -1, -1, -1,
                                     -1, -1, +1, +0, +1, -1, -1, -1, -1, -1, -1, +0, +1, +1));
    check_select(dataset, 9,
                 make_tensor<scalar_t>(make_dims(10, 4, 1, 1), -1, +1, -1, +1, +0, +0, +0, +0, +1, +1, +1, +1, +2, +4,
                                       +8, +16, +3, +9, +27, +81, +4, +16, +64, +256, +5, +25, +125, +625, +6, +36,
                                       +216, 36 * 36, +7, +49, 49 * 7, 49 * 49, +8, +64, 64 * 8, 64 * 64));
    check_select(dataset, 10,
                 make_tensor<scalar_t>(make_dims(10, 4, 1, 1), -2, +4, -8, +16, Na, Na, Na, Na, +0, +0, +0, +0, Na, Na,
                                       Na, Na, +2, +4, +8, +16, Na, Na, Na, Na, +4, +16, +64, +256, Na, Na, Na, Na, +6,
                                       +36, +216, 36 * 36, Na, Na, Na, Na));
    check_select(dataset, 11,
                 make_tensor<scalar_t>(make_dims(10, 4, 1, 1), -3, +9, -27, +81, Na, Na, Na, Na, Na, Na, Na, Na, +0, +0,
                                       +0, +0, Na, Na, Na, Na, Na, Na, Na, Na, +3, +9, +27, +81, Na, Na, Na, Na, Na, Na,
                                       Na, Na, +6, +36, +216, 36 * 36));

    check_flatten(
        dataset,
        make_tensor<scalar_t>(
            make_dims(10, 27), -1, -1, -1, +1, +1, +1, +1, -1, -1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1, -2, +4, -8,
            +16, -3, +9, -27, +81, +1, Na, Na, -1, Na, Na, -1, +1, +1, Na, Na, Na, Na, Na, Na, +0, +0, +0, +0, Na, Na,
            Na, Na, Na, Na, Na, Na, +1, +1, Na, -1, -1, Na, +1, -1, -1, -1, +1, +1, Na, Na, Na, +1, +1, +1, +1, +0, +0,
            +0, +0, Na, Na, Na, Na, +1, Na, +1, -1, Na, -1, -1, +1, -1, Na, Na, Na, -1, +1, +1, +2, +4, +8, 16, Na, Na,
            Na, Na, +0, +0, +0, +0, +1, +1, Na, -1, -1, Na, +1, -1, +1, -1, +1, -1, Na, Na, Na, +3, +9, 27, 81, +2, +4,
            +8, 16, Na, Na, Na, Na, +1, Na, Na, -1, Na, Na, -1, +1, -1, Na, Na, Na, Na, Na, Na, +4, 16, 64, 256, Na, Na,
            Na, Na, Na, Na, Na, Na, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1, +1, -1, +1, -1, +1, +5, 25, 125, 625, +4,
            16, 64, 256, +3, +9, +27, +81, +1, Na, Na, -1, Na, Na, -1, +1, +1, Na, Na, Na, Na, Na, Na, +6, 36, 36 * 6,
            36 * 36, Na, Na, Na, Na, Na, Na, Na, Na, +1, +1, Na, -1, -1, Na, +1, -1, -1, -1, +1, +1, Na, Na, Na, +7, 49,
            49 * 7, 49 * 49, +6, 36, 36 * 6, 36 * 36, Na, Na, Na, Na, +1, Na, +1, -1, Na, -1, -1, +1, -1, Na, Na, Na,
            -1, +1, +1, +8, 64, 64 * 8, 64 * 64, Na, Na, Na, Na, +6, +36, 36 * 6, 36 * 36),
        make_indices(0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11));
}

UTEST_CASE(structured)
{
    const auto datasource = make_datasource(10, string_t::npos);

    auto dataset = dataset_t{datasource};
    add_generator<elemwise_generator_t<struct_to_mclass_t>>(dataset);
    add_generator<elemwise_generator_t<struct_to_struct_t>>(dataset, make_indices(8, 9, 10));
    add_generator<elemwise_generator_t<struct_to_sclass_t>>(dataset);
    add_generator<elemwise_generator_t<struct_to_scalar_t>>(dataset);

    UTEST_REQUIRE_EQUAL(dataset.features(), 12);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"feature(struct0)"}.mclass(strings_t{"<10", "<30", "<20"}));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"feature(struct1)"}.mclass(strings_t{"<10", "<30", "<20"}));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"feature(struct2)"}.mclass(strings_t{"<10", "<30", "<20"}));
    UTEST_CHECK_EQUAL(dataset.feature(3),
                      feature_t{"feature(struct0)"}.scalar(feature_type::float64, make_dims(2, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(4),
                      feature_t{"feature(struct1)"}.scalar(feature_type::float64, make_dims(2, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(5),
                      feature_t{"feature(struct2)"}.scalar(feature_type::float64, make_dims(2, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(6), feature_t{"feature(struct0)"}.sclass(strings_t{"<10", ">=10"}));
    UTEST_CHECK_EQUAL(dataset.feature(7), feature_t{"feature(struct1)"}.sclass(strings_t{"<10", ">=10"}));
    UTEST_CHECK_EQUAL(dataset.feature(8), feature_t{"feature(struct2)"}.sclass(strings_t{"<10", ">=10"}));
    UTEST_CHECK_EQUAL(dataset.feature(9), feature_t{"feature(struct0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(10), feature_t{"feature(struct1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(11), feature_t{"feature(struct2)"}.scalar(feature_type::float64));

    check_select(dataset, 0,
                 make_tensor<int8_t>(make_dims(10, 3), +1, +1, +1, +1, +1, +1, +1, +1, +1, +0, +1, +1, +0, +1, +1, +0,
                                     +1, +0, +0, +1, +0, +0, +1, +0, +0, +0, +0, +0, +0, +0));
    check_select(dataset, 1,
                 make_tensor<int8_t>(make_dims(10, 3), +1, +1, +1, -1, -1, -1, +0, +1, +1, -1, -1, -1, +0, +1, +0, -1,
                                     -1, -1, +0, +0, +0, -1, -1, -1, +0, +0, +0, -1, -1, -1));
    check_select(dataset, 2,
                 make_tensor<int8_t>(make_dims(10, 3), +1, +1, +1, -1, -1, -1, -1, -1, -1, +0, +1, +1, -1, -1, -1, -1,
                                     -1, -1, +0, +1, +1, -1, -1, -1, -1, -1, -1, +0, +1, +0));
    check_select(dataset, 3,
                 make_tensor<scalar_t>(make_dims(10, 2, 1, 1), 1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29,
                                       30, 33, 34, 37, 38));
    check_select(dataset, 4,
                 make_tensor<scalar_t>(make_dims(10, 2, 1, 1), 1, 2, N, N, 13, 14, N, Na, 25, 26, Na, Na, 37, 38, Na,
                                       Na, 49, 50, Na, Na));
    check_select(dataset, 5,
                 make_tensor<scalar_t>(make_dims(10, 2, 1, 1), 1, 2, N, N, N, Na, 10, 11, Na, Na, Na, Na, 19, 20, Na,
                                       Na, Na, Na, 28, 29));
    check_select(dataset, 6, make_tensor<int32_t>(make_dims(10), +0, +0, +0, +1, +1, +1, +1, +1, +1, +1));
    check_select(dataset, 7, make_tensor<int32_t>(make_dims(10), +0, -1, +1, -1, +1, -1, +1, -1, +1, -1));
    check_select(dataset, 8, make_tensor<int32_t>(make_dims(10), +0, -1, -1, +1, -1, -1, +1, -1, -1, +1));
    check_select(dataset, 9, make_tensor<scalar_t>(make_dims(10), 1, 5, +9, 13, 17, 21, 25, 29, 33, 37));
    check_select(dataset, 10, make_tensor<scalar_t>(make_dims(10), 1, N, 13, Na, 25, Na, 37, Na, 49, Na));
    check_select(dataset, 11, make_tensor<scalar_t>(make_dims(10), 1, N, Na, 10, Na, Na, 19, Na, Na, 28));

    check_flatten(dataset,
                  make_tensor<scalar_t>(
                      make_dims(10, 21), +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +2, +1, +2, +1, +2, +1, +1, +1, +1, +1,
                      +1, +1, +1, +1, +0, +0, +0, +0, +0, +0, +5, +6, Na, Na, Na, Na, +1, Na, Na, +5, Na, Na, +1, +1,
                      +1, -1, +1, +1, +0, +0, +0, +9, 10, 13, 14, Na, Na, +1, -1, Na, +9, 13, Na, -1, +1, +1, +0, +0,
                      +0, -1, +1, +1, 13, 14, Na, Na, 10, 11, -1, Na, -1, 13, Na, 10, -1, +1, +1, -1, +1, -1, +0, +0,
                      +0, 17, 18, 25, 26, Na, Na, -1, -1, Na, 17, 25, Na, -1, +1, -1, +0, +0, +0, +0, +0, +0, 21, 22,
                      Na, Na, Na, Na, -1, Na, Na, 21, Na, Na, -1, +1, -1, -1, -1, -1, -1, +1, +1, 25, 26, 37, 38, 19,
                      20, -1, -1, -1, 25, 37, 19, -1, +1, -1, +0, +0, +0, +0, +0, +0, 29, 30, Na, Na, Na, Na, -1, Na,
                      Na, 29, Na, Na, -1, -1, -1, -1, -1, -1, +0, +0, +0, 33, 34, 49, 50, Na, Na, -1, -1, Na, 33, 49,
                      Na, -1, -1, -1, +0, +0, +0, -1, +1, -1, 37, 38, Na, Na, 28, 29, -1, Na, -1, 37, Na, 28),
                  make_indices(0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 11));
}

UTEST_END_MODULE()
