#include <fixture/generator.h>
#include <fixture/generator_datasource.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;

namespace
{
auto make_dataset(const datasource_t& datasource)
{
    auto dataset = dataset_t{datasource};
    add_generator<sclass_identity_generator_t>(dataset);
    add_generator<mclass_identity_generator_t>(dataset);
    add_generator<scalar_identity_generator_t>(dataset);
    add_generator<struct_identity_generator_t>(dataset);
    return dataset;
}

auto keep(const tensor2d_t& flatten, const indices_t& columns_to_keep)
{
    const auto [samples, columns] = flatten.dims();

    tensor2d_t    tensor(samples, columns_to_keep.size());
    tensor_size_t column2 = 0;
    for (const auto column : columns_to_keep)
    {
        tensor.matrix().col(column2++) = flatten.matrix().col(column);
    }
    return tensor;
}

auto remove(const tensor2d_t& flatten, const indices_t& columns_to_remove)
{
    const auto [samples, columns] = flatten.dims();

    const auto* const begin = std::begin(columns_to_remove);
    const auto* const end   = std::end(columns_to_remove);

    tensor2d_t tensor(samples, columns - columns_to_remove.size());
    for (tensor_size_t column = 0, column2 = 0; column < columns; ++column)
    {
        if (std::find(begin, end, column) == end)
        {
            tensor.matrix().col(column2++) = flatten.matrix().col(column);
        }
    }
    return tensor;
}

auto drop(const tensor2d_t& flatten, const indices_t& columns)
{
    tensor2d_t tensor = flatten;
    for (const auto column : columns)
    {
        tensor.matrix().array().col(column) = NaN;
    }
    return tensor;
}

auto expected_sclass0()
{
    return make_features()[2];
}

auto expected_sclass1()
{
    return make_features()[3];
}

auto expected_sclass2()
{
    return make_features()[4];
}

auto expected_mclass0()
{
    return make_features()[0];
}

auto expected_mclass1()
{
    return make_features()[1];
}

auto expected_scalar0()
{
    return make_features()[5];
}

auto expected_scalar1()
{
    return make_features()[6];
}

auto expected_scalar2()
{
    return make_features()[7];
}

auto expected_struct0()
{
    return make_features()[8];
}

auto expected_struct1()
{
    return make_features()[9];
}

auto expected_struct2()
{
    return make_features()[10];
}

auto expected_select_sclass0()
{
    return make_tensor<int32_t>(make_dims(10), +2, -1, +1, -1, +0, -1, +2, -1, +1, -1);
}

auto expected_select_sclass1()
{
    return make_tensor<int32_t>(make_dims(10), +1, +0, +1, +0, +1, +0, +1, +0, +1, +0);
}

auto expected_select_sclass2()
{
    return make_tensor<int32_t>(make_dims(10), +0, -1, +0, -1, +0, -1, +0, -1, +0, -1);
}

auto expected_select_mclass0()
{
    return make_tensor<int8_t>(make_dims(10, 3), +0, +1, +1, +1, +0, +0, +0, +1, +0, +1, +0, +0, +0, +1, +0, +1, +0, +0,
                               +0, +1, +1, +1, +0, +0, +0, +1, +0, +1, +0, +0);
}

auto expected_select_mclass1()
{
    return make_tensor<int8_t>(make_dims(10, 4), +0, +1, +1, +0, -1, -1, -1, -1, +0, +1, +0, +0, -1, -1, -1, -1, +0, +1,
                               +0, +0, -1, -1, -1, -1, +0, +1, +1, +0, -1, -1, -1, -1, +0, +1, +0, +0, -1, -1, -1, -1);
}

auto expected_select_scalar0()
{
    return make_tensor<scalar_t>(make_dims(10), -1, +0, +1, +2, +3, +4, +5, +6, +7, +8);
}

auto expected_select_scalar1()
{
    return make_tensor<scalar_t>(make_dims(10), -2, Na, +0, Na, +2, Na, +4, Na, +6, Na);
}

auto expected_select_scalar2()
{
    return make_tensor<scalar_t>(make_dims(10), -3, Na, Na, +0, Na, Na, +3, Na, Na, +6);
}

auto expected_select_struct0()
{
    return make_tensor<scalar_t>(make_dims(10, 1, 2, 2), +1, +0, +0, +0, +2, +1, +1, +1, +3, +2, +2, +2, +4, +3, +3, +3,
                                 +5, +4, +4, +4, +6, +5, +5, +5, +7, +6, +6, +6, +8, +7, +7, +7, +9, +8, +8, +8, +10,
                                 +9, +9, +9);
}

auto expected_select_struct1()
{
    return make_tensor<scalar_t>(make_dims(10, 2, 1, 3), +1, +0, +0, +0, +0, +0, Na, Na, Na, Na, Na, Na, +3, +2, +2, +2,
                                 +2, +2, Na, Na, Na, Na, Na, Na, +5, +4, +4, +4, +4, +4, Na, Na, Na, Na, Na, Na, +7, +6,
                                 +6, +6, +6, +6, Na, Na, Na, Na, Na, Na, +9, +8, +8, +8, +8, +8, Na, Na, Na, Na, Na,
                                 Na);
}

auto expected_select_struct2()
{
    return make_tensor<scalar_t>(make_dims(10, 3, 1, 1), +1, +0, +0, Na, Na, Na, Na, Na, Na, +4, +3, +3, Na, Na, Na, Na,
                                 Na, Na, +7, +6, +6, Na, Na, Na, Na, Na, Na, +10, +9, +9);
}

auto expected_flatten()
{
    return make_tensor<scalar_t>(
        make_dims(10, 27), -1, -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0,
        +0, +1, +0, +0, Na, Na, +1, Na, +1, -1, -1, Na, Na, Na, Na, +0, Na, Na, +2, +1, +1, +1, Na, Na, Na, Na, Na, Na,
        Na, Na, Na, -1, +1, -1, +1, -1, +1, -1, -1, +1, -1, -1, +1, +0, Na, +3, +2, +2, +2, +3, +2, +2, +2, +2, +2, Na,
        Na, Na, Na, Na, +1, Na, +1, -1, -1, Na, Na, Na, Na, +2, Na, +0, +4, +3, +3, +3, Na, Na, Na, Na, Na, Na, +4, +3,
        +3, +1, -1, -1, +1, -1, +1, -1, -1, +1, -1, -1, +3, +2, Na, +5, +4, +4, +4, +5, +4, +4, +4, +4, +4, Na, Na, Na,
        Na, Na, +1, Na, +1, -1, -1, Na, Na, Na, Na, +4, Na, Na, +6, +5, +5, +5, Na, Na, Na, Na, Na, Na, Na, Na, Na, -1,
        -1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +5, +4, +3, +7, +6, +6, +6, +7, +6, +6, +6, +6, +6, +7, +6, +6, Na, Na,
        +1, Na, +1, -1, -1, Na, Na, Na, Na, +6, Na, Na, +8, +7, +7, +7, Na, Na, Na, Na, Na, Na, Na, Na, Na, -1, +1, -1,
        +1, -1, +1, -1, -1, +1, -1, -1, +7, +6, Na, +9, +8, +8, +8, +9, +8, +8, +8, +8, +8, Na, Na, Na, Na, Na, +1, Na,
        +1, -1, -1, Na, Na, Na, Na, +8, Na, +6, 10, +9, +9, +9, Na, Na, Na, Na, Na, Na, 10, +9, +9);
}
} // namespace

UTEST_BEGIN_MODULE(test_generator_identity)

UTEST_CASE(empty)
{
    const auto datasource = make_datasource(10, string_t::npos);
    const auto dataset    = dataset_t{datasource};

    UTEST_CHECK_EQUAL(dataset.columns(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
}

UTEST_CASE(unsupervised)
{
    const auto datasource = make_datasource(10, string_t::npos);
    const auto dataset    = make_dataset(datasource);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::unsupervised);

    UTEST_REQUIRE_EQUAL(dataset.features(), 11);
    UTEST_CHECK_EQUAL(dataset.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(dataset.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(dataset.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(dataset.feature(3), expected_mclass0());
    UTEST_CHECK_EQUAL(dataset.feature(4), expected_mclass1());
    UTEST_CHECK_EQUAL(dataset.feature(5), expected_scalar0());
    UTEST_CHECK_EQUAL(dataset.feature(6), expected_scalar1());
    UTEST_CHECK_EQUAL(dataset.feature(7), expected_scalar2());
    UTEST_CHECK_EQUAL(dataset.feature(8), expected_struct0());
    UTEST_CHECK_EQUAL(dataset.feature(9), expected_struct1());
    UTEST_CHECK_EQUAL(dataset.feature(10), expected_struct2());

    check_select(dataset, 0, expected_select_sclass0());
    check_select(dataset, 1, expected_select_sclass1());
    check_select(dataset, 2, expected_select_sclass2());
    check_select(dataset, 3, expected_select_mclass0());
    check_select(dataset, 4, expected_select_mclass1());
    check_select(dataset, 5, expected_select_scalar0());
    check_select(dataset, 6, expected_select_scalar1());
    check_select(dataset, 7, expected_select_scalar2());
    check_select(dataset, 8, expected_select_struct0());
    check_select(dataset, 9, expected_select_struct1());
    check_select(dataset, 10, expected_select_struct2());
    check_select_stats(dataset, make_indices(0, 1, 2), make_indices(3, 4), make_indices(5, 6, 7),
                       make_indices(8, 9, 10));

    const auto expected_flatten = ::expected_flatten();
    const auto expected_columns =
        make_indices(0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10);

    check_flatten(dataset, expected_flatten, expected_columns);

    dataset.drop(0);
    check_flatten(dataset, drop(expected_flatten, make_indices(0, 1)), expected_columns, true);

    dataset.drop(2);
    check_flatten(dataset, drop(expected_flatten, make_indices(0, 1, 3)), expected_columns, true);

    dataset.undrop();
    check_flatten(dataset, expected_flatten, expected_columns);

    check_flatten_stats(
        dataset, make_indices(5, 5, 10, 5, 10, 10, 10, 5, 5, 5, 5, 10, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(27), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -3, +1, +0, +0, +0, +1, +0, +0,
                              +0, +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(27), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +8, +6, +6, 10, +9, +9, +9, +9, +8, +8,
                              +8, +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(27), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0,
                              4.0, 4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(27), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.027650354097, 3.162277660168,
                              3.872983346207, 3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097,
                              3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
                              3.162277660168, 3.872983346207, 3.872983346207, 3.872983346207));

    check_targets_stats(dataset, indices_t{}, tensor1d_t{}, tensor1d_t{}, tensor1d_t{}, tensor1d_t{});

    UTEST_CHECK_EQUAL(dataset.target(), feature_t{});
    UTEST_CHECK_EQUAL(dataset.target_dims(), make_dims(0, 0, 0));

    const auto samples = arange(0, dataset.samples());

    auto iterator = flatten_iterator_t{dataset, samples};
    iterator.batch(128);
    UTEST_CHECK_THROW(iterator.loop([&](tensor_range_t, size_t, tensor4d_cmap_t) {}), std::runtime_error);
}

UTEST_CASE(sclassification)
{
    const auto datasource = make_datasource(10, 3U);
    const auto dataset    = make_dataset(datasource);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::sclassification);

    UTEST_REQUIRE_EQUAL(dataset.features(), 10);
    UTEST_CHECK_EQUAL(dataset.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(dataset.feature(1), expected_sclass2());
    UTEST_CHECK_EQUAL(dataset.feature(2), expected_mclass0());
    UTEST_CHECK_EQUAL(dataset.feature(3), expected_mclass1());
    UTEST_CHECK_EQUAL(dataset.feature(4), expected_scalar0());
    UTEST_CHECK_EQUAL(dataset.feature(5), expected_scalar1());
    UTEST_CHECK_EQUAL(dataset.feature(6), expected_scalar2());
    UTEST_CHECK_EQUAL(dataset.feature(7), expected_struct0());
    UTEST_CHECK_EQUAL(dataset.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(dataset.feature(9), expected_struct2());

    check_select(dataset, 0, expected_select_sclass0());
    check_select(dataset, 1, expected_select_sclass2());
    check_select(dataset, 2, expected_select_mclass0());
    check_select(dataset, 3, expected_select_mclass1());
    check_select(dataset, 4, expected_select_scalar0());
    check_select(dataset, 5, expected_select_scalar1());
    check_select(dataset, 6, expected_select_scalar2());
    check_select(dataset, 7, expected_select_struct0());
    check_select(dataset, 8, expected_select_struct1());
    check_select(dataset, 9, expected_select_struct2());
    check_select_stats(dataset, make_indices(0, 1), make_indices(2, 3), make_indices(4, 5, 6), make_indices(7, 8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(2));
    const auto expected_columns =
        make_indices(0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(dataset, expected_flatten, expected_columns);

    dataset.drop(0);
    check_flatten(dataset, drop(expected_flatten, make_indices(0, 1)), expected_columns, true);

    dataset.undrop();
    check_flatten(dataset, expected_flatten, expected_columns);

    check_flatten_stats(
        dataset, make_indices(5, 5, 5, 10, 10, 10, 5, 5, 5, 5, 10, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0,
                              +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +8, +6, +6, 10, +9, +9, +9, +9, +8, +8, +8,
                              +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0,
                              4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.027650354097, 3.162277660168,
                              3.872983346207, 3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097,
                              3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
                              3.162277660168, 3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(dataset, expected_sclass1(), make_dims(2, 1, 1),
                  make_tensor<scalar_t>(make_dims(10, 2, 1, 1), -1, +1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1, +1,
                                        +1, -1, -1, +1, +1, -1));
    check_targets_stats(dataset, make_indices(10, 10), make_tensor<scalar_t>(make_dims(2), 0.0, 0.0),
                        make_tensor<scalar_t>(make_dims(2), 0.0, 0.0), make_tensor<scalar_t>(make_dims(2), 0.0, 0.0),
                        make_tensor<scalar_t>(make_dims(2), 0.0, 0.0));
}

UTEST_CASE(mclassification)
{
    const auto datasource = make_datasource(10, 0U);
    const auto dataset    = make_dataset(datasource);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::mclassification);

    UTEST_REQUIRE_EQUAL(dataset.features(), 10);
    UTEST_CHECK_EQUAL(dataset.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(dataset.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(dataset.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(dataset.feature(3), expected_mclass1());
    UTEST_CHECK_EQUAL(dataset.feature(4), expected_scalar0());
    UTEST_CHECK_EQUAL(dataset.feature(5), expected_scalar1());
    UTEST_CHECK_EQUAL(dataset.feature(6), expected_scalar2());
    UTEST_CHECK_EQUAL(dataset.feature(7), expected_struct0());
    UTEST_CHECK_EQUAL(dataset.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(dataset.feature(9), expected_struct2());

    check_select(dataset, 0, expected_select_sclass0());
    check_select(dataset, 1, expected_select_sclass1());
    check_select(dataset, 2, expected_select_sclass2());
    check_select(dataset, 3, expected_select_mclass1());
    check_select(dataset, 4, expected_select_scalar0());
    check_select(dataset, 5, expected_select_scalar1());
    check_select(dataset, 6, expected_select_scalar2());
    check_select(dataset, 7, expected_select_struct0());
    check_select(dataset, 8, expected_select_struct1());
    check_select(dataset, 9, expected_select_struct2());
    check_select_stats(dataset, make_indices(0, 1, 2), make_indices(3), make_indices(4, 5, 6), make_indices(7, 8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(4, 5, 6));
    const auto expected_columns = make_indices(0, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(dataset, expected_flatten, expected_columns);

    dataset.drop(3);
    check_flatten(dataset, drop(expected_flatten, make_indices(4, 5, 6, 7)), expected_columns, true);

    dataset.undrop();
    check_flatten(dataset, expected_flatten, expected_columns);

    check_flatten_stats(
        dataset, make_indices(5, 5, 10, 5, 5, 5, 5, 5, 10, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(24), 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0, +0, +0,
                              +1, +0, +0),
        make_tensor<scalar_t>(make_dims(24), 0, 0, 0, 0, 0, 0, 0, 0, +8, +6, +6, 10, +9, +9, +9, +9, +8, +8, +8, +8, +8,
                              10, +9, +9),
        make_tensor<scalar_t>(make_dims(24), 0, 0, 0, 0, 0, 0, 0, 0, 3.5, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0, 4.0,
                              4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(24), 0, 0, 0, 0, 0, 0, 0, 0, 3.027650354097, 3.162277660168, 3.872983346207,
                              3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097, 3.162277660168,
                              3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
                              3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(dataset, expected_mclass0(), make_dims(3, 1, 1),
                  keep(::expected_flatten(), make_indices(4, 5, 6)).reshape(10, 3, 1, 1));
    check_targets_stats(dataset, make_indices(10, 10, 10), make_tensor<scalar_t>(make_dims(3), 0.0, 0.0, 0.0),
                        make_tensor<scalar_t>(make_dims(3), 0.0, 0.0, 0.0),
                        make_tensor<scalar_t>(make_dims(3), 0.0, 0.0, 0.0),
                        make_tensor<scalar_t>(make_dims(3), 0.0, 0.0, 0.0));
}

UTEST_CASE(regression)
{
    const auto datasource = make_datasource(10, 5U);
    const auto dataset    = make_dataset(datasource);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::regression);

    UTEST_REQUIRE_EQUAL(dataset.features(), 10);
    UTEST_CHECK_EQUAL(dataset.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(dataset.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(dataset.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(dataset.feature(3), expected_mclass0());
    UTEST_CHECK_EQUAL(dataset.feature(4), expected_mclass1());
    UTEST_CHECK_EQUAL(dataset.feature(5), expected_scalar1());
    UTEST_CHECK_EQUAL(dataset.feature(6), expected_scalar2());
    UTEST_CHECK_EQUAL(dataset.feature(7), expected_struct0());
    UTEST_CHECK_EQUAL(dataset.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(dataset.feature(9), expected_struct2());

    check_select(dataset, 0, expected_select_sclass0());
    check_select(dataset, 1, expected_select_sclass1());
    check_select(dataset, 2, expected_select_sclass2());
    check_select(dataset, 3, expected_select_mclass0());
    check_select(dataset, 4, expected_select_mclass1());
    check_select(dataset, 5, expected_select_scalar1());
    check_select(dataset, 6, expected_select_scalar2());
    check_select(dataset, 7, expected_select_struct0());
    check_select(dataset, 8, expected_select_struct1());
    check_select(dataset, 9, expected_select_struct2());
    check_select_stats(dataset, make_indices(0, 1, 2), make_indices(3, 4), make_indices(5, 6), make_indices(7, 8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(11));
    const auto expected_columns =
        make_indices(0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(dataset, expected_flatten, expected_columns);

    dataset.drop(0);
    check_flatten(dataset, drop(expected_flatten, make_indices(0, 1)), expected_columns, true);

    dataset.undrop();
    check_flatten(dataset, expected_flatten, expected_columns);

    check_flatten_stats(
        dataset, make_indices(5, 5, 10, 5, 10, 10, 10, 5, 5, 5, 5, 5, 4, 10, 10, 10, 10, 5, 5, 5, 5, 5, 5, 4, 4, 4),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, -3, +1, +0, +0, +0, +1, +0, +0, +0,
                              +0, +0, +1, +0, +0),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +6, +6, 10, +9, +9, +9, +9, +8, +8, +8,
                              +8, +8, 10, +9, +9),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.0, 1.5, 5.5, 4.5, 4.5, 4.5, 5.0, 4.0,
                              4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(26), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.162277660168, 3.872983346207,
                              3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097, 3.162277660168,
                              3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
                              3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(dataset, expected_scalar0(), make_dims(1, 1, 1),
                  keep(::expected_flatten(), make_indices(11)).reshape(10, 1, 1, 1));
    check_targets_stats(dataset, make_indices(10), make_tensor<scalar_t>(make_dims(1), -1),
                        make_tensor<scalar_t>(make_dims(1), 8), make_tensor<scalar_t>(make_dims(1), 3.5),
                        make_tensor<scalar_t>(make_dims(1), 3.027650354097));
}

UTEST_CASE(mvregression)
{
    const auto datasource = make_datasource(10, 8U);
    const auto dataset    = make_dataset(datasource);
    UTEST_REQUIRE_EQUAL(dataset.type(), task_type::regression);

    UTEST_REQUIRE_EQUAL(dataset.features(), 10);
    UTEST_CHECK_EQUAL(dataset.feature(0), expected_sclass0());
    UTEST_CHECK_EQUAL(dataset.feature(1), expected_sclass1());
    UTEST_CHECK_EQUAL(dataset.feature(2), expected_sclass2());
    UTEST_CHECK_EQUAL(dataset.feature(3), expected_mclass0());
    UTEST_CHECK_EQUAL(dataset.feature(4), expected_mclass1());
    UTEST_CHECK_EQUAL(dataset.feature(5), expected_scalar0());
    UTEST_CHECK_EQUAL(dataset.feature(6), expected_scalar1());
    UTEST_CHECK_EQUAL(dataset.feature(7), expected_scalar2());
    UTEST_CHECK_EQUAL(dataset.feature(8), expected_struct1());
    UTEST_CHECK_EQUAL(dataset.feature(9), expected_struct2());

    check_select(dataset, 0, expected_select_sclass0());
    check_select(dataset, 1, expected_select_sclass1());
    check_select(dataset, 2, expected_select_sclass2());
    check_select(dataset, 3, expected_select_mclass0());
    check_select(dataset, 4, expected_select_mclass1());
    check_select(dataset, 5, expected_select_scalar0());
    check_select(dataset, 6, expected_select_scalar1());
    check_select(dataset, 7, expected_select_scalar2());
    check_select(dataset, 8, expected_select_struct1());
    check_select(dataset, 9, expected_select_struct2());
    check_select_stats(dataset, make_indices(0, 1, 2), make_indices(3, 4), make_indices(5, 6, 7), make_indices(8, 9));

    const auto expected_flatten = remove(::expected_flatten(), make_indices(14, 15, 16, 17));
    const auto expected_columns = make_indices(0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9);

    check_flatten(dataset, expected_flatten, expected_columns);

    dataset.drop(1);
    check_flatten(dataset, drop(expected_flatten, make_indices(2)), expected_columns, true);

    dataset.undrop();
    check_flatten(dataset, expected_flatten, expected_columns);

    check_flatten_stats(dataset, make_indices(5, 5, 10, 5, 10, 10, 10, 5, 5, 5, 5, 10, 5, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4),
                        make_tensor<scalar_t>(make_dims(23), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -3, +1, +0, +0,
                                              +0, +0, +0, +1, +0, +0),
                        make_tensor<scalar_t>(make_dims(23), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +8, +6, +6, +9, +8, +8,
                                              +8, +8, +8, 10, +9, +9),
                        make_tensor<scalar_t>(make_dims(23), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.5, 2.0, 1.5, 5.0, 4.0,
                                              4.0, 4.0, 4.0, 4.0, 5.5, 4.5, 4.5),
                        make_tensor<scalar_t>(make_dims(23), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.027650354097,
                                              3.162277660168, 3.872983346207, 3.162277660168, 3.162277660168,
                                              3.162277660168, 3.162277660168, 3.162277660168, 3.162277660168,
                                              3.872983346207, 3.872983346207, 3.872983346207));

    check_targets(dataset, expected_struct0(), make_dims(1, 2, 2),
                  keep(::expected_flatten(), make_indices(14, 15, 16, 17)).reshape(10, 1, 2, 2));
    check_targets_stats(
        dataset, make_indices(10, 10, 10, 10), make_tensor<scalar_t>(make_dims(4), 1, 0, 0, 0),
        make_tensor<scalar_t>(make_dims(4), 10, 9, 9, 9), make_tensor<scalar_t>(make_dims(4), 5.5, 4.5, 4.5, 4.5),
        make_tensor<scalar_t>(make_dims(4), 3.027650354097, 3.027650354097, 3.027650354097, 3.027650354097));
}

UTEST_END_MODULE()
