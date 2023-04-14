#include <nano/function/benchmark/styblinski_tang.h>

using namespace nano;

function_styblinski_tang_t::function_styblinski_tang_t(tensor_size_t dims)
    : function_t("styblinski-tang", dims)
{
    convex(convexity::no);
    smooth(smoothness::yes);
}

rfunction_t function_styblinski_tang_t::clone() const
{
    return std::make_unique<function_styblinski_tang_t>(*this);
}

scalar_t function_styblinski_tang_t::do_vgrad(const vector_t& x, vector_t* gx) const
{
    if (gx != nullptr)
    {
        *gx = 4 * x.array().cube() - 32 * x.array() + 5;
    }

    return (x.array().square().square() - 16 * x.array().square() + 5 * x.array()).sum();
}

rfunction_t function_styblinski_tang_t::make(tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_styblinski_tang_t>(dims);
}
