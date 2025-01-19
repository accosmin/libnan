#include <fixture/generator.h>
#include <fixture/generator_datasource.h>
#include <nano/generator/select.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_generator_select)

UTEST_CASE(select_scalar)
{
    const auto dataset = make_datasource(10, string_t::npos);
    {
        const auto mapping = select_scalar(dataset);
        const auto expected_mapping =
            make_tensor<tensor_size_t>(make_dims(3, 5), 5, 0, 1, 1, 1, 6, 0, 1, 1, 1, 7, 0, 1, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping          = select_scalar(dataset, make_indices(0, 1, 3, 6));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 5), 6, 0, 1, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
}

UTEST_CASE(select_struct)
{
    const auto dataset = make_datasource(10, string_t::npos);
    {
        const auto mapping = select_struct(dataset);
        const auto expected_mapping =
            make_tensor<tensor_size_t>(make_dims(3, 5), 8, 0, 1, 2, 2, 9, 0, 2, 1, 3, 10, 0, 3, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping          = select_struct(dataset, make_indices(2, 5, 8));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 5), 8, 0, 1, 2, 2);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping          = select_struct(dataset, make_indices(2, 4));
        const auto expected_mapping = feature_mapping_t{0, 5};
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
}

UTEST_CASE(select_sclass)
{
    const auto dataset = make_datasource(10, string_t::npos);
    {
        const auto mapping = select_sclass(dataset);
        const auto expected_mapping =
            make_tensor<tensor_size_t>(make_dims(3, 5), 2, 3, 1, 1, 1, 3, 2, 1, 1, 1, 4, 2, 1, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping          = select_sclass(dataset, make_indices(0, 1, 2));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(1, 5), 2, 3, 1, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
}

UTEST_CASE(select_mclass)
{
    const auto dataset = make_datasource(10, string_t::npos);
    {
        const auto mapping          = select_mclass(dataset);
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(2, 5), 0, 3, 1, 1, 1, 1, 4, 1, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
    {
        const auto mapping          = select_mclass(dataset, make_indices(0, 1, 2, 3, 4));
        const auto expected_mapping = make_tensor<tensor_size_t>(make_dims(2, 5), 0, 3, 1, 1, 1, 1, 4, 1, 1, 1);
        UTEST_CHECK_EQUAL(mapping, expected_mapping);
    }
}

UTEST_END_MODULE()
