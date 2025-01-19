#pragma once

#include <nano/function/quadratic.h>
#include <nano/solver.h>

using namespace nano;

namespace nano::gsample
{
struct sampler_t
{
    explicit sampler_t(tensor_size_t n);

    template <class tmatrix>
    void descent(quadratic_program_t& program, const tmatrix& W, vector_t& g, const logger_t& logger)
    {
        const auto G = m_G.slice(0, m_psize);
        program.reset(G * W * G.transpose());

        const auto x0    = vector_t::constant(m_psize, 1.0 / static_cast<scalar_t>(m_psize));
        const auto state = m_solver->minimize(program, x0, logger);
        // FIXME: this cannot be guaranteed, better to show a warning!
        // assert(state.status() == solver_status::converged);
        g = G.transpose() * state.x();
        g = W * g;
    }

    static quadratic_program_t make_program(tensor_size_t p);

    // attributes
    matrix_t      m_X;        ///< buffer of sample points (p, n)
    matrix_t      m_G;        ///< buffer of sample gradients (p, n)
    tensor_size_t m_psize{0}; ///< current number of samples
    rsolver_t     m_solver;   ///< solver for the quadratic program to compute the sample gradient
};

class fixed_sampler_t final : public sampler_t
{
public:
    explicit fixed_sampler_t(tensor_size_t n);

    static auto str() { return "gs"; }

    void sample(const solver_state_t& state, scalar_t epsilon);

    template <class tmatrix>
    void descent(const tmatrix& W, vector_t& g, const logger_t& logger)
    {
        sampler_t::descent(m_program, W, g, logger);
    }

private:
    // attributes
    quadratic_program_t m_program;
};

class adaptive_sampler_t final : public sampler_t
{
public:
    explicit adaptive_sampler_t(tensor_size_t n);

    static auto str() { return "ags"; }

    void sample(const solver_state_t& state, scalar_t epsilon);

    template <class tmatrix>
    void descent(const tmatrix& W, vector_t& g, const logger_t& logger)
    {
        auto program = make_program(m_psize);
        sampler_t::descent(program, W, g, logger);
    }
};
} // namespace nano::gsample
