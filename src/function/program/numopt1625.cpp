#include <function/program/numopt1625.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>

using namespace nano;

quadratic_program_numopt1625_t::quadratic_program_numopt1625_t(const tensor_size_t dims)
    : quadratic_program_t("numopt1625", matrix_t{matrix_t::zero(dims, dims)}, vector_t::zero(dims))
{
    const auto x0 = make_random_vector<scalar_t>(dims);
    const auto Q  = matrix_t::identity(dims, dims);
    const auto c  = -x0;
    const auto l  = make_random_vector<scalar_t>(dims);
    const auto u  = l.array() + 0.1;

    reset(Q, c);
    optimum(x0.array().max(l.array()).min(u.array()));

    critical(l <= variable());
    critical(variable() <= u);
}

rfunction_t quadratic_program_numopt1625_t::clone() const
{
    return std::make_unique<quadratic_program_numopt1625_t>(*this);
}

rfunction_t quadratic_program_numopt1625_t::make(const tensor_size_t                  dims,
                                                 [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<quadratic_program_numopt1625_t>(dims);
}
