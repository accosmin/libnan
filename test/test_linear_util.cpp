#include "fixture/linear.h"
#include "fixture/loss.h"
#include <nano/core/reduce.h>
#include <nano/linear/accumulator.h>
#include <nano/linear/util.h>

using namespace nano;

UTEST_BEGIN_MODULE(test_linear_util)

UTEST_CASE(accumulator)
{
    const auto fill_accumulator = [](linear::accumulator_t& accumulator, const scalar_t value)
    {
        accumulator.m_vm1 = value;
        accumulator.m_gb1.full(value);
        accumulator.m_gW1.full(value);
    };

    const auto make_accumulators = [&]()
    {
        auto accumulators = std::vector<linear::accumulator_t>(3U, linear::accumulator_t{3, 2});
        fill_accumulator(accumulators[0], 1);
        fill_accumulator(accumulators[1], 2);
        fill_accumulator(accumulators[2], 3);
        return accumulators;
    };

    {
        auto accumulators = make_accumulators();

        const auto& accumulator0 = sum_reduce(accumulators, 6);
        UTEST_CHECK_CLOSE(accumulator0.m_vm1, 6.0 / 6.0, 1e-12);
        UTEST_CHECK_CLOSE(accumulator0.m_gb1, make_full_tensor<scalar_t>(make_dims(2), 6.0 / 6.0), 1e-12);
        UTEST_CHECK_CLOSE(accumulator0.m_gW1, make_full_tensor<scalar_t>(make_dims(2, 3), 6.0 / 6.0), 1e-12);
    }
}

UTEST_CASE(predict)
{
    const auto epsilon = epsilon1<scalar_t>();

    const auto bias    = make_random_tensor<scalar_t>(make_dims(3));
    const auto weights = make_random_tensor<scalar_t>(make_dims(3, 5));
    const auto inputs  = make_random_tensor<scalar_t>(make_dims(11, 5));

    tensor4d_t outputs;
    linear::predict(inputs, weights, bias, outputs);

    for (tensor_size_t sample = 0; sample < inputs.size<0>(); ++sample)
    {
        UTEST_CHECK_CLOSE(outputs.vector(sample), weights.matrix() * inputs.vector(sample) + bias.vector(), epsilon);
    }
}

UTEST_CASE(evaluate)
{
    const auto datasource = make_linear_datasource(20, 3, 4);
    const auto dataset    = make_dataset(datasource);
    const auto loss       = make_loss("mse");

    const auto samples         = arange(0, dataset.samples());
    const auto expected_values = make_full_tensor<scalar_t>(make_dims(2, samples.size()), 0.0);

    for (const auto batch : {1, 2, 3, 4})
    {
        const auto& bias    = datasource.bias();
        const auto& weights = datasource.weights();
        const auto  values  = linear::evaluate(dataset, samples, *loss, weights, bias, batch);

        UTEST_CHECK_CLOSE(values, expected_values, 1e-12);
    }
}

UTEST_END_MODULE()
