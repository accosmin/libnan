#include <function/benchmark/quadratic.h>
#include <nano/function/util.h>

using namespace nano;

function_quadratic_t::function_quadratic_t(const tensor_size_t dims)
    : function_t("quadratic", dims)
    , m_a(make_random_vector<scalar_t>(dims, -1.0, +1.0, seed_t{42}))
{
    // NB: generate random positive semi-definite matrix to keep the function convex
    const auto A = make_random_matrix<scalar_t>(dims, dims, -1.0, +1.0, seed_t{42});
    m_A          = matrix_t::identity(dims, dims) + A * A.transpose();

    convex(convexity::yes);
    smooth(smoothness::yes);
    strong_convexity(::strong_convexity(m_A));
}

rfunction_t function_quadratic_t::clone() const
{
    return std::make_unique<function_quadratic_t>(*this);
}

scalar_t function_quadratic_t::do_vgrad(vector_cmap_t x, vector_map_t gx) const
{
    const auto a = m_a.vector();
    const auto A = m_A.matrix();

    if (gx.size() == x.size())
    {
        gx = a + A * x.vector();
    }

    return x.dot(a + 0.5 * (A * x.vector()));
}

rfunction_t function_quadratic_t::make(const tensor_size_t dims, tensor_size_t) const
{
    return std::make_unique<function_quadratic_t>(dims);
}
