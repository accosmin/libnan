#include <mutex>
#include <nano/core/numeric.h>
#include <nano/lsearchk/backtrack.h>
#include <nano/lsearchk/cgdescent.h>
#include <nano/lsearchk/fletcher.h>
#include <nano/lsearchk/lemarechal.h>
#include <nano/lsearchk/morethuente.h>

using namespace nano;

lsearchk_t::lsearchk_t()
{
    register_parameter(parameter_t::make_scalar_pair("lsearchk::tolerance", 0, LT, 1e-4, LT, 0.1, LT, 1));
    register_parameter(parameter_t::make_integer("lsearchk::max_iterations", 1, LE, 100, LE, 1000));
}

lsearchk_factory_t& lsearchk_t::all()
{
    static auto manager = lsearchk_factory_t{};
    const auto  op      = []()
    {
        manager.add<lsearchk_fletcher_t>("fletcher", "Fletcher (strong Wolfe conditions)");
        manager.add<lsearchk_backtrack_t>("backtrack", "backtrack using cubic interpolation (Armijo conditions)");
        manager.add<lsearchk_cgdescent_t>("cgdescent", "CG-DESCENT (regular and approximate Wolfe conditions)");
        manager.add<lsearchk_lemarechal_t>("lemarechal", "LeMarechal (regular Wolfe conditions)");
        manager.add<lsearchk_morethuente_t>("morethuente", "More&Thuente (strong Wolfe conditions)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

static auto make_state0(const solver_state_t& state)
{
    auto state0 = state;
    state0.t    = 0;
    return state0;
}

bool lsearchk_t::get(solver_state_t& state, scalar_t t) const
{
    const auto max_iterations = parameter("lsearchk::max_iterations").value<int>();

    // check descent direction
    if (!state.has_descent())
    {
        return false;
    }

    // adjust the initial step length if it produces an invalid state
    const auto state0 = make_state0(state);
    assert(state0.t < epsilon0<scalar_t>());

    t = std::isfinite(t) ? std::clamp(t, stpmin(), scalar_t(1)) : scalar_t(1);
    for (int i = 0; i < max_iterations; ++i)
    {
        const auto ok = state.update(state0, t);
        log(state0, state);

        if (!ok)
        {
            t *= 0.5;
        }
        else
        {
            break;
        }
    }

    // line-search step length
    // NB: some line-search algorithms (see CGDESCENT) allow a small increase
    //     in the function value when close to numerical precision!
    return get(state0, state) && state;
}
