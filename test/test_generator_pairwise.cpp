#include <fixture/generator.h>
#include <fixture/generator_datasource.h>
#include <nano/generator/pairwise.h>

using namespace nano;

template <class tinput, class tgenerated>
class NANO_PUBLIC tester_t : public tinput, public tgenerated
{
public:
    template <class... targs>
    explicit tester_t(targs... args)
        : tinput("gg", std::forward<targs>(args)...)
    {
    }
};

class NANO_PUBLIC scalar_scalar_to_scalar_t : public tester_t<pairwise_input_scalar_scalar_t, generated_scalar_t>
{
public:
    using tester_t::tester_t;

    feature_t feature(const tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "sum"); }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values1, const auto& values2)
        { return static_cast<scalar_t>(values1(0)) + static_cast<scalar_t>(values2(0)); };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC scalar_scalar_to_struct_t : public tester_t<pairwise_input_scalar_scalar_t, generated_struct_t>
{
public:
    using tester_t::tester_t;

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_struct_feature(ifeature, "pow", make_dims(3, 1, 1));
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{3};
        const auto process = [=](const auto& values1, const auto& values2, auto&& structured)
        {
            const auto value1 = static_cast<scalar_t>(values1(0));
            const auto value2 = static_cast<scalar_t>(values2(0));
            structured(0)     = value1 * value1;
            structured(1)     = value1 * value2;
            structured(2)     = value2 * value2;
        };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC scalar_scalar_to_sclass_t : public tester_t<pairwise_input_scalar_scalar_t, generated_sclass_t>
{
public:
    using tester_t::tester_t;

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_sclass_feature(ifeature, "sign", strings_t{"neg", "pos"});
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& values1, const auto& values2)
        {
            const auto value1 = static_cast<scalar_t>(values1(0));
            const auto value2 = static_cast<scalar_t>(values2(0));
            return (value1 < 0.0 || value2 < 0.0) ? 0 : 1;
        };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC scalar_scalar_to_mclass_t : public tester_t<pairwise_input_scalar_scalar_t, generated_mclass_t>
{
public:
    using tester_t::tester_t;

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_mclass_feature(ifeature, "mod", strings_t{"mod2", "mod3"});
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{2};
        const auto process = [=](const auto& values1, const auto& values2, auto&& mclass)
        {
            const auto value1 = static_cast<int>(static_cast<scalar_t>(values1(0)));
            const auto value2 = static_cast<int>(static_cast<scalar_t>(values2(0)));
            mclass(0)         = (value1 + value2) % 2 == 0 ? 0x01 : 0x00;
            mclass(1)         = (value1 + value2) % 3 == 0 ? 0x01 : 0x00;
        };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC sclass_sclass_to_scalar_t : public tester_t<pairwise_input_sclass_sclass_t, generated_scalar_t>
{
public:
    using tester_t::tester_t;

    feature_t feature(const tensor_size_t ifeature) const override { return make_scalar_feature(ifeature, "sum"); }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{1};
        const auto process = [=](const auto& value1, const auto& value2)
        { return static_cast<scalar_t>(value1) + static_cast<scalar_t>(value2); };
        return std::make_tuple(process, colsize);
    }
};

class NANO_PUBLIC sclass_sclass_to_struct_t : public tester_t<pairwise_input_sclass_sclass_t, generated_struct_t>
{
public:
    using tester_t::tester_t;

    feature_t feature(const tensor_size_t ifeature) const override
    {
        return make_struct_feature(ifeature, "pow", make_dims(3, 1, 1));
    }

    static auto process(const tensor_size_t)
    {
        const auto colsize = tensor_size_t{3};
        const auto process = [=](const auto& value1, const auto& value2, auto&& structured)
        {
            const auto v1 = static_cast<scalar_t>(value1);
            const auto v2 = static_cast<scalar_t>(value2);
            structured(0) = v1 * v1;
            structured(1) = v1 * v2;
            structured(2) = v2 * v2;
        };
        return std::make_tuple(process, colsize);
    }
};

UTEST_BEGIN_MODULE(test_generator_pairwise)

UTEST_CASE(scalar_scalar)
{
    const auto datasource = make_datasource(10, string_t::npos);

    auto dataset = dataset_t{datasource};
    add_generator<pairwise_generator_t<scalar_scalar_to_scalar_t>>(dataset);
    add_generator<pairwise_generator_t<scalar_scalar_to_struct_t>>(dataset, make_indices(6));
    add_generator<pairwise_generator_t<scalar_scalar_to_sclass_t>>(dataset);
    add_generator<pairwise_generator_t<scalar_scalar_to_mclass_t>>(dataset, make_indices(6), make_indices(6, 7));

    UTEST_REQUIRE_EQUAL(dataset.features(), 15);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"sum(scalar0,scalar0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"sum(scalar0,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"sum(scalar0,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_t{"sum(scalar1,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(4), feature_t{"sum(scalar1,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(5), feature_t{"sum(scalar2,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(6),
                      feature_t{"pow(scalar1,scalar1)"}.scalar(feature_type::float64, make_dims(3, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(7), feature_t{"sign(scalar0,scalar0)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(8), feature_t{"sign(scalar0,scalar1)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(9), feature_t{"sign(scalar0,scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(10), feature_t{"sign(scalar1,scalar1)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(11), feature_t{"sign(scalar1,scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(12), feature_t{"sign(scalar2,scalar2)"}.sclass(strings_t{"neg", "pos"}));
    UTEST_CHECK_EQUAL(dataset.feature(13), feature_t{"mod(scalar1,scalar1)"}.mclass(strings_t{"mod2", "mod3"}));
    UTEST_CHECK_EQUAL(dataset.feature(14), feature_t{"mod(scalar1,scalar2)"}.mclass(strings_t{"mod2", "mod3"}));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(10), -2, 0, 2, 4, 6, 8, 10, 12, 14, 16));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(10), -3, N, 1, N, 5, N, +9, Na, 13, Na));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(10), -4, N, N, 2, N, N, +8, Na, Na, 14));
    check_select(dataset, 3, make_tensor<scalar_t>(make_dims(10), -4, N, 0, N, 4, N, +8, Na, 12, Na));
    check_select(dataset, 4, make_tensor<scalar_t>(make_dims(10), -5, N, N, N, N, N, +7, Na, Na, Na));
    check_select(dataset, 5, make_tensor<scalar_t>(make_dims(10), -6, N, N, 0, N, N, +6, Na, Na, 12));
    check_select(dataset, 6,
                 make_tensor<scalar_t>(make_dims(10, 3, 1, 1), 4, 4, 4, N, N, N, 0, 0, 0, N, N, N, 4, 4, 4, N, N, N, 16,
                                       16, 16, N, N, N, 36, 36, 36, N, N, N));
    check_select(dataset, 7, make_tensor<int32_t>(make_dims(10), +0, +1, +1, +1, +1, +1, +1, +1, +1, +1));
    check_select(dataset, 8, make_tensor<int32_t>(make_dims(10), +0, -1, +1, -1, +1, -1, +1, -1, +1, -1));
    check_select(dataset, 9, make_tensor<int32_t>(make_dims(10), +0, -1, -1, +1, -1, -1, +1, -1, -1, +1));
    check_select(dataset, 10, make_tensor<int32_t>(make_dims(10), +0, -1, +1, -1, +1, -1, +1, -1, +1, -1));
    check_select(dataset, 11, make_tensor<int32_t>(make_dims(10), +0, -1, -1, -1, -1, -1, +1, -1, -1, -1));
    check_select(dataset, 12, make_tensor<int32_t>(make_dims(10), +0, -1, -1, +1, -1, -1, +1, -1, -1, +1));
    check_select(dataset, 13,
                 make_tensor<int8_t>(make_dims(10, 2), +1, +0, -1, -1, +1, +1, -1, -1, +1, +0, -1, -1, +1, +0, -1, -1,
                                     +1, +1, -1, -1));
    check_select(dataset, 14,
                 make_tensor<int8_t>(make_dims(10, 2), +0, +0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, +0, +0, -1, -1,
                                     -1, -1, -1, -1));

    dataset.drop(0);
    check_flatten(dataset,
                  make_tensor<scalar_t>(make_dims(10, 19), Na, -3, -4, -4, -5, -6, +4, +4, +4, +1, +1, +1, +1, +1, +1,
                                        +1, -1, -1, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na,
                                        Na, Na, Na, Na, +1, Na, +0, Na, Na, +0, +0, +0, -1, -1, Na, -1, Na, Na, +1, +1,
                                        Na, Na, Na, Na, +2, Na, Na, +0, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na,
                                        Na, Na, +5, Na, +4, Na, Na, +4, +4, +4, -1, -1, Na, -1, Na, Na, +1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na,
                                        +9, +8, +8, +7, +6, 16, 16, 16, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na, 13, Na,
                                        12, Na, Na, 36, 36, 36, -1, -1, Na, -1, Na, Na, +1, +1, Na, Na, Na, Na, 14, Na,
                                        Na, 12, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na, Na),
                  make_indices(0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14), true);

    dataset.drop(6);
    check_flatten(dataset,
                  make_tensor<scalar_t>(make_dims(10, 19), Na, -3, -4, -4, -5, -6, Na, Na, Na, +1, +1, +1, +1, +1, +1,
                                        +1, -1, -1, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na,
                                        Na, Na, Na, Na, +1, Na, +0, Na, Na, Na, Na, Na, -1, -1, Na, -1, Na, Na, +1, +1,
                                        Na, Na, Na, Na, +2, Na, Na, +0, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na,
                                        Na, Na, +5, Na, +4, Na, Na, Na, Na, Na, -1, -1, Na, -1, Na, Na, +1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na,
                                        +9, +8, +8, +7, +6, Na, Na, Na, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1, Na, Na,
                                        Na, Na, Na, Na, Na, Na, Na, -1, Na, Na, Na, Na, Na, Na, Na, Na, Na, Na, 13, Na,
                                        12, Na, Na, Na, Na, Na, -1, -1, Na, -1, Na, Na, +1, +1, Na, Na, Na, Na, 14, Na,
                                        Na, 12, Na, Na, Na, -1, Na, -1, Na, Na, -1, Na, Na, Na, Na),
                  make_indices(0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 14, 14), true);
}

UTEST_CASE(sclass_sclass)
{
    const auto datasource = make_datasource(10, string_t::npos);

    auto dataset = dataset_t{datasource};
    add_generator<pairwise_generator_t<sclass_sclass_to_scalar_t>>(dataset);
    add_generator<pairwise_generator_t<sclass_sclass_to_struct_t>>(dataset, make_indices(2, 3), make_indices(4));

    UTEST_REQUIRE_EQUAL(dataset.features(), 8);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"sum(sclass0,sclass0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"sum(sclass0,sclass1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"sum(sclass0,sclass2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_t{"sum(sclass1,sclass1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(4), feature_t{"sum(sclass1,sclass2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(5), feature_t{"sum(sclass2,sclass2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(6),
                      feature_t{"pow(sclass0,sclass2)"}.scalar(feature_type::float64, make_dims(3, 1, 1)));
    UTEST_CHECK_EQUAL(dataset.feature(7),
                      feature_t{"pow(sclass1,sclass2)"}.scalar(feature_type::float64, make_dims(3, 1, 1)));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(10), 4, N, 2, N, 0, N, 4, N, 2, N));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(10), 3, N, 2, N, 1, N, 3, N, 2, N));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(10), 2, N, 1, N, 0, N, 2, N, 1, N));
    check_select(dataset, 3, make_tensor<scalar_t>(make_dims(10), 2, 0, 2, 0, 2, 0, 2, 0, 2, 0));
    check_select(dataset, 4, make_tensor<scalar_t>(make_dims(10), 1, N, 1, N, 1, N, 1, N, 1, N));
    check_select(dataset, 5, make_tensor<scalar_t>(make_dims(10), 0, N, 0, N, 0, N, 0, N, 0, N));
    check_select(dataset, 6,
                 make_tensor<scalar_t>(make_dims(10, 3, 1, 1), 4, 0, 0, N, N, N, 1, 0, 0, N, N, N, 0, 0, 0, N, N, N, 4,
                                       0, 0, N, N, N, 1, 0, 0, N, N, N));
    check_select(dataset, 7,
                 make_tensor<scalar_t>(make_dims(10, 3, 1, 1), 1, 0, 0, N, N, N, 1, 0, 0, N, N, N, 1, 0, 0, N, N, N, 1,
                                       0, 0, N, N, N, 1, 0, 0, N, N, N));

    check_flatten(dataset,
                  make_tensor<scalar_t>(make_dims(10, 12), 4, 3, 2, 2, 1, 0, 4, 0, 0, 1, 0, 0, N, N, N, 0, N, N, N, N,
                                        N, N, N, N, 2, 2, 1, 2, 1, 0, 1, 0, 0, 1, 0, 0, N, N, N, 0, N, N, N, N, N, N, N,
                                        N, 0, 1, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, N, N, N, 0, N, N, N, N, N, N, N, N, 4, 3,
                                        2, 2, 1, 0, 4, 0, 0, 1, 0, 0, N, N, N, 0, N, N, N, N, N, N, N, N, 2, 2, 1, 2, 1,
                                        0, 1, 0, 0, 1, 0, 0, N, N, N, 0, N, N, N, N, N, N, N, N),
                  make_indices(0, 1, 2, 3, 4, 5, 6, 6, 6, 7, 7, 7));
}

UTEST_END_MODULE()
