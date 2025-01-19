#include <fixture/generator.h>
#include <fixture/generator_datasource.h>
#include <nano/generator/pairwise_product.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_generator_product)

UTEST_CASE(empty)
{
    const auto datasource = make_datasource(10, string_t::npos);
    const auto dataset    = dataset_t{datasource};

    UTEST_CHECK_EQUAL(dataset.columns(), 0);
    UTEST_CHECK_EQUAL(dataset.features(), 0);
}

UTEST_CASE(product_all)
{
    const auto datasource = make_datasource(10, string_t::npos);
    auto       dataset    = dataset_t{datasource};

    UTEST_CHECK_NOTHROW(dataset.add<pairwise_product_generator_t>());

    UTEST_REQUIRE_EQUAL(dataset.features(), 6);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"product(scalar0,scalar0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"product(scalar0,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"product(scalar0,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_t{"product(scalar1,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(4), feature_t{"product(scalar1,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(5), feature_t{"product(scalar2,scalar2)"}.scalar(feature_type::float64));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(10), 1, 0, 1, 4, 9, 16, 25, 36, 49, 64));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(10), 2, N, 0, N, 6, Na, 20, Na, 42, Na));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(10), 3, N, N, 0, N, Na, 15, Na, Na, 48));
    check_select(dataset, 3, make_tensor<scalar_t>(make_dims(10), 4, N, 0, N, 4, Na, 16, Na, 36, Na));
    check_select(dataset, 4, make_tensor<scalar_t>(make_dims(10), 6, N, N, N, N, Na, 12, Na, Na, Na));
    check_select(dataset, 5, make_tensor<scalar_t>(make_dims(10), 9, N, N, 0, N, Na, +9, Na, Na, 36));

    check_flatten(dataset,
                  make_tensor<scalar_t>(make_dims(10, 6), 1, 2, 3, 4, 6, 9, 0, N, N, N, N, N, 1, 0, N, 0, N, N, 4, N, 0,
                                        N, N, 0, 9, 6, N, 4, N, N, 16, Na, Na, Na, Na, Na, 25, 20, 15, 16, 12, +9, 36,
                                        Na, Na, Na, Na, Na, 49, 42, Na, 36, Na, Na, 64, Na, 48, Na, Na, 36),
                  make_indices(0, 1, 2, 3, 4, 5));
}

UTEST_CASE(product_some1)
{
    const auto datasource = make_datasource(10, string_t::npos);
    auto       dataset    = dataset_t{datasource};

    UTEST_CHECK_NOTHROW(dataset.add<pairwise_product_generator_t>(make_indices(5, 6)));

    UTEST_REQUIRE_EQUAL(dataset.features(), 3);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"product(scalar0,scalar0)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"product(scalar0,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"product(scalar1,scalar1)"}.scalar(feature_type::float64));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(10), 1, 0, 1, 4, 9, 16, 25, 36, 49, 64));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(10), 2, N, 0, N, 6, Na, 20, Na, 42, Na));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(10), 4, N, 0, N, 4, Na, 16, Na, 36, Na));

    check_flatten(dataset,
                  make_tensor<scalar_t>(make_dims(10, 3), 1, 2, 4, 0, N, N, 1, 0, 0, 4, N, N, 9, 6, 4, 16, Na, Na, 25,
                                        20, 16, 36, Na, Na, 49, 42, 36, 64, Na, Na),
                  make_indices(0, 1, 2));
}

UTEST_CASE(product_some2)
{
    const auto datasource = make_datasource(10, string_t::npos);
    auto       dataset    = dataset_t{datasource};

    UTEST_CHECK_NOTHROW(dataset.add<pairwise_product_generator_t>(make_indices(5, 6), make_indices(6, 7)));

    UTEST_REQUIRE_EQUAL(dataset.features(), 4);
    UTEST_CHECK_EQUAL(dataset.feature(0), feature_t{"product(scalar0,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(1), feature_t{"product(scalar0,scalar2)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(2), feature_t{"product(scalar1,scalar1)"}.scalar(feature_type::float64));
    UTEST_CHECK_EQUAL(dataset.feature(3), feature_t{"product(scalar1,scalar2)"}.scalar(feature_type::float64));

    check_select(dataset, 0, make_tensor<scalar_t>(make_dims(10), 2, N, 0, N, 6, Na, 20, Na, 42, Na));
    check_select(dataset, 1, make_tensor<scalar_t>(make_dims(10), 3, N, N, 0, N, Na, 15, Na, Na, 48));
    check_select(dataset, 2, make_tensor<scalar_t>(make_dims(10), 4, N, 0, N, 4, Na, 16, Na, 36, Na));
    check_select(dataset, 3, make_tensor<scalar_t>(make_dims(10), 6, N, N, N, N, Na, 12, Na, Na, Na));

    check_flatten(dataset,
                  make_tensor<scalar_t>(make_dims(10, 4), 2, 3, 4, 6, N, N, N, N, 0, N, 0, N, N, 0, N, N, 6, N, 4, N,
                                        Na, Na, Na, Na, 20, 15, 16, 12, Na, Na, Na, Na, 42, Na, 36, Na, Na, 48, Na, Na),
                  make_indices(0, 1, 2, 3));
}

UTEST_END_MODULE()
