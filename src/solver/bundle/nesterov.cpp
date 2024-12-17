#include <solver/bundle/nesterov.h>

using namespace nano;

nesterov_sequence1_t::nesterov_sequence1_t(const solver_state_t& state)
    : nesterov_sequence_t<nesterov_sequence1_t>(state)
{
}

std::tuple<scalar_t, scalar_t> nesterov_sequence1_t::make_alpha_beta()
{
    const auto curr  = lambda();
    const auto next  = update();
    const auto alpha = (curr - 1.0) / next;
    const auto beta  = 0.0;
    return std::make_tuple(alpha, beta);
}

nesterov_sequence2_t::nesterov_sequence2_t(const solver_state_t& state)
    : nesterov_sequence_t<nesterov_sequence2_t>(state)
{
}

std::tuple<scalar_t, scalar_t> nesterov_sequence2_t::make_alpha_beta()
{
    const auto curr  = lambda();
    const auto next  = update();
    const auto alpha = (curr - 1.0) / next;
    const auto beta  = curr / next;
    return std::make_tuple(alpha, beta);
}
