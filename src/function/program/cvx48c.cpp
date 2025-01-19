#include <function/program/cvx48c.h>
#include <nano/critical.h>
#include <nano/function/bounds.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx48c_t::linear_program_cvx48c_t(const tensor_size_t dims)
    : linear_program_t("cvx48c", vector_t::zero(dims))
{
    const auto c = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto l = make_random_vector<scalar_t>(dims, -1.0, +1.0);
    const auto u = make_random_vector<scalar_t>(dims, +1.0, +3.0);

    reset(c);
    optimum(l.array() * c.array().max(0.0).sign() - u.array() * c.array().min(0.0).sign());

    critical(l <= variable());
    critical(variable() <= u);
}

rfunction_t linear_program_cvx48c_t::clone() const
{
    return std::make_unique<linear_program_cvx48c_t>(*this);
}

rfunction_t linear_program_cvx48c_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48c_t>(dims);
}
