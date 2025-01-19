#include <function/program/cvx48b.h>
#include <nano/core/scat.h>
#include <nano/critical.h>
#include <nano/function/cuts.h>

using namespace nano;

linear_program_cvx48b_t::linear_program_cvx48b_t(const tensor_size_t dims, const scalar_t lambda)
    : linear_program_t(scat("cvx48b[lambda=", lambda, "]"), vector_t::zero(dims))
{
    const auto a = make_random_vector<scalar_t>(dims, +1.0, +2.0);
    const auto b = urand<scalar_t>(-1.0, +1.0);
    const auto c = lambda * a;

    reset(c);
    optimum(lambda * b);

    critical(lambda <= 0.0);
    critical((a * variable()) <= b);
}

rfunction_t linear_program_cvx48b_t::clone() const
{
    return std::make_unique<linear_program_cvx48b_t>(*this);
}

rfunction_t linear_program_cvx48b_t::make(const tensor_size_t dims, [[maybe_unused]] const tensor_size_t summands) const
{
    return std::make_unique<linear_program_cvx48b_t>(dims);
}
