#include <nano/core/strutil.h>
#include <nano/function/elastic_net.h>

using namespace nano;

static auto make_suffix(scalar_t alpha1, scalar_t alpha2)
{
    if (alpha1 == 0.0)
    {
        return "Ridge";
    }
    else
    {
        return alpha2 == 0.0 ? "Lasso": "ElasticNet";
    }
}

static auto make_size(tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2});
}

static auto make_inputs(tensor_size_t dims)
{
    return std::max(dims, tensor_size_t{2}) - 1;
}

static auto make_outputs(tensor_size_t)
{
    return tensor_size_t{1};
}

template <typename tloss>
function_enet_t<tloss>::function_enet_t(tensor_size_t dims, scalar_t alpha1, scalar_t alpha2, tensor_size_t summands) :
    benchmark_function_t(scat(tloss::basename, "+", make_suffix(alpha1, alpha2)), ::make_size(dims)),
    tloss(summands, make_outputs(dims), make_inputs(dims)),
    m_alpha1(alpha1),
    m_alpha2(alpha2)
{
    convex(tloss::convex);
    smooth(m_alpha1 == 0.0 && tloss::smooth);
    this->summands(summands);
}

template <typename tloss>
scalar_t function_enet_t<tloss>::vgrad(const vector_t& x, vector_t* gx) const
{
    const auto inputs = this->inputs();
    const auto targets = this->targets();
    const auto outputs = this->outputs(x);

    const auto fx = tloss::vgrad(inputs, outputs, targets, gx);
    return fx + regularize(x, gx);
}

template <typename tloss>
scalar_t function_enet_t<tloss>::vgrad(const vector_t& x, tensor_size_t summand, vector_t* gx) const
{
    const auto inputs = this->inputs(summand);
    const auto targets = this->targets(summand);
    const auto outputs = this->outputs(x, summand);

    const auto fx = tloss::vgrad(inputs, outputs, targets, gx);
    return fx + regularize(x, gx);
}

template <typename tloss>
scalar_t function_enet_t<tloss>::regularize(const vector_t& x, vector_t* gx) const
{
    const auto w = tloss::make_w(x).matrix();

    if (gx != nullptr)
    {
        auto gw = tloss::make_w(*gx).matrix();

        gw.array() += m_alpha1 * w.array().sign();
        // cppcheck-suppress unreadVariable
        gw += m_alpha2 * w;
    }

    return m_alpha1 * w.template lpNorm<1>() + 0.5 * m_alpha2 * w.squaredNorm();
}

template <typename tloss>
rfunction_t function_enet_t<tloss>::make(tensor_size_t dims, tensor_size_t summands) const
{
    return std::make_unique<function_enet_t<tloss>>(dims, m_alpha1, m_alpha2, summands);
}

template class nano::function_enet_t<nano::loss_mse_t>;
template class nano::function_enet_t<nano::loss_mae_t>;
template class nano::function_enet_t<nano::loss_hinge_t>;
template class nano::function_enet_t<nano::loss_cauchy_t>;
template class nano::function_enet_t<nano::loss_logistic_t>;
