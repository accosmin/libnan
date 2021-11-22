#include <nano/function/axis_ellipsoid.h>

using namespace nano;

function_axis_ellipsoid_t::function_axis_ellipsoid_t(tensor_size_t dims) :
    benchmark_function_t("Axis Parallel Hyper-Ellipsoid", dims),
    m_bias(vector_t::LinSpaced(dims, scalar_t(1), scalar_t(dims)))
{
    convex(true);
    smooth(true);
}

scalar_t function_axis_ellipsoid_t::vgrad(const vector_t& x, vector_t* gx, vgrad_config_t) const
{
    if (gx != nullptr)
    {
        *gx = 2 * x.array() * m_bias.array();
    }

    return (x.array().square() * m_bias.array()).sum();
}

rfunction_t function_axis_ellipsoid_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_axis_ellipsoid_t>(dims);
}