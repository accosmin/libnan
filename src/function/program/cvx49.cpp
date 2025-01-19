#include <function/program/cvx49.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx49_t::linear_program_cvx49_t(const tensor_size_t dims)
    : linear_program_t("cvx49", vector_t::zero(dims))
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, -0.0);
    const auto A = matrix_t::identity(dims, dims);
    const auto b = make_random_vector<scalar_t>(dims, -1.0, +1.0);

    reset(c);
    optimum(b);

    critical((A * variable()) <= b);
}

rfunction_t linear_program_cvx49_t::clone() const
{
    return std::make_unique<linear_program_cvx49_t>(*this);
}

rfunction_t linear_program_cvx49_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx49_t>(dims);
}
