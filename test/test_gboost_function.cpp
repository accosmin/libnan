#include <fixture/dataset.h>
#include <fixture/loss.h>
#include <fixture/solver.h>
#include <nano/gboost/function.h>
#include <nano/generator/elemwise_identity.h>

using namespace nano;
using namespace nano::gboost;

namespace
{
class fixture_datasource_t final : public datasource_t
{
public:
    explicit fixture_datasource_t(const tensor_size_t samples = 100, const tensor_size_t isize = 3,
                                  const tensor_size_t tsize = 2, const tensor_size_t groups = 3)
        : datasource_t("fixture")
        , m_groups(groups)
        , m_samples(samples)
        , m_idims(make_dims(isize, 1, 1))
        , m_tdims(make_dims(tsize, 1, 1))
        , m_targets(make_dims(samples, tsize, 1, 1))
    {
    }

    rdatasource_t clone() const override { return std::make_unique<fixture_datasource_t>(*this); }

    vector_t bias(const indices_t& samples) const
    {
        const auto targets = m_targets.indexed(samples);
        const auto tmatrix = targets.reshape(targets.size<0>(), -1).matrix();
        return tmatrix.colwise().mean();
    }

    tensor4d_t targets(const indices_t& samples) const { return m_targets.indexed(samples); }

    cluster_t cluster(const indices_t& samples) const
    {
        cluster_t cluster(this->samples(), m_groups);
        for (const auto sample : samples)
        {
            if (sample % 2 > 0)
            {
                cluster.assign(sample, sample % m_groups);
            }
        }
        return cluster;
    }

    auto groups() const { return m_groups; }

    const auto& scale() const { return m_scale; }

    const auto& outputs() const { return m_outputs; }

    const auto& targets() const { return m_targets; }

    const auto& woutputs() const { return m_woutputs; }

private:
    void do_load() override
    {
        const auto features = features_t{feature_t{"inputs"}.scalar(feature_type::float32, m_idims),
                                         feature_t{"target"}.scalar(feature_type::float64, m_tdims)};

        resize(m_samples, features, 1U);

        m_scale    = make_random_vector<scalar_t>(m_groups, 1.1, 2.2);
        m_outputs  = make_random_tensor<scalar_t>(cat_dims(m_samples, m_tdims));
        m_woutputs = make_random_tensor<scalar_t>(cat_dims(m_samples, m_tdims));

        tensor3d_t inputs(m_idims);

        for (tensor_size_t sample = 0; sample < m_samples; ++sample)
        {
            const auto group = sample % m_groups;

            inputs.random();
            m_targets.vector(sample) = m_outputs.vector(sample) + m_scale(group) * m_woutputs.vector(sample);

            set(sample, 0, inputs);
            set(sample, 1, m_targets.tensor(sample));
        }
    }

    // attributes
    vector_t        m_scale;        ///<
    tensor4d_t      m_outputs;      ///<
    tensor4d_t      m_woutputs;     ///<
    tensor_size_t   m_groups{1};    ///<
    tensor_size_t   m_samples{100}; ///< total number of samples to generate (train + validation + test)
    tensor3d_dims_t m_idims{};      ///< dimension of an input sample
    tensor3d_dims_t m_tdims{};      ///< dimension of a target/output sample
    tensor4d_t      m_targets;      ///<
};

auto make_datasource(const tensor_size_t samples = 100, const tensor_size_t isize = 3, const tensor_size_t tsize = 2,
                     const tensor_size_t groups = 3)
{
    auto datasource = fixture_datasource_t{samples, isize, tsize, groups};
    UTEST_CHECK_NOTHROW(datasource.load());
    UTEST_CHECK_EQUAL(datasource.samples(), samples);
    return datasource;
}

void check_optimum(const function_t& function, const vector_t& expected_optimum)
{
    const auto solver = make_solver();
    const auto state  = check_minimize(*solver, function, vector_t::zero(function.size()));
    UTEST_CHECK_CLOSE(state.x(), expected_optimum, 1e+2 * solver->parameter("solver::epsilon").value<scalar_t>());
}

template <class ttmatrix, class tomatrix>
void check_value(const function_t& function, const ttmatrix& tmatrix, const tomatrix& omatrix,
                 const scalar_t epsilon = 1e-12)
{
    const auto values = 0.5 * (tmatrix - omatrix).array().square().rowwise().sum();

    UTEST_CHECK_CLOSE(function(make_full_vector<scalar_t>(function.size(), 0.0)), values.mean(), epsilon);
}
} // namespace

UTEST_BEGIN_MODULE(test_gboost_function)

UTEST_CASE(bias)
{
    const auto loss       = make_loss();
    const auto datasource = make_datasource(100);
    const auto dataset    = make_dataset(datasource);

    for (const auto& samples : {arange(0, 100), arange(10, 60), arange(0, 50), arange(10, 100)})
    {
        const auto iterator = targets_iterator_t{dataset, samples};
        const auto bias     = datasource.bias(samples);
        const auto targets  = datasource.targets(samples);
        const auto tmatrix  = targets.reshape(targets.size<0>(), -1).matrix();
        const auto omatrix  = matrix_t::zero(tmatrix.rows(), tmatrix.cols());

        const auto function = bias_function_t{iterator, *loss};

        UTEST_CHECK_EQUAL(function.size(), 2);
        check_gradient(function, 10);
        check_convexity(function, 10);
        check_value(function, tmatrix, omatrix);
        check_optimum(function, bias);
    }
}

UTEST_CASE(scale)
{
    const auto loss       = make_loss();
    const auto datasource = make_datasource(50);
    const auto dataset    = make_dataset(datasource);

    // NB: the outputs are provided for all available samples
    const auto  all_samples = arange(0, datasource.samples());
    const auto& scale       = datasource.scale();
    const auto  cluster     = datasource.cluster(all_samples);
    const auto& outputs     = datasource.outputs();
    const auto& woutputs    = datasource.woutputs();
    const auto& targets     = datasource.targets();
    const auto  tmatrix     = targets.reshape(targets.size<0>(), -1).matrix();
    const auto  omatrix     = outputs.reshape(tmatrix.rows(), tmatrix.cols()).matrix();

    // ... but the scaling is only computed for the training samples
    for (const auto& samples : {arange(0, 50), arange(10, 40), arange(0, 40), arange(10, 50)})
    {
        const auto iterator = targets_iterator_t{dataset, samples};

        const auto function = scale_function_t{iterator, *loss, cluster, outputs, woutputs};

        UTEST_CHECK_EQUAL(function.size(), datasource.groups());
        check_gradient(function, 10);
        check_convexity(function, 10);
        if (samples.size() == datasource.samples())
        {
            check_value(function, tmatrix, omatrix);
        }
        check_optimum(function, scale);
    }
}

UTEST_CASE(grads)
{
    const auto loss       = make_loss();
    const auto datasource = make_datasource(10);
    const auto dataset    = make_dataset(datasource);

    const auto  all_samples = arange(0, datasource.samples());
    const auto  iterator    = targets_iterator_t{dataset, all_samples};
    const auto& targets     = datasource.targets();
    const auto  tmatrix     = targets.reshape(targets.size<0>(), -1).matrix();
    const auto  omatrix     = matrix_t::zero(tmatrix.rows(), tmatrix.cols());

    const auto function = grads_function_t{iterator, *loss};

    UTEST_CHECK_EQUAL(function.size(), all_samples.size() * 2);
    check_gradient(function, 10);
    check_convexity(function, 10);
    check_value(function, tmatrix, omatrix);
    check_optimum(function, targets.vector());
}

UTEST_END_MODULE()
