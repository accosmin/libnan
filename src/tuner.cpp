#include <mutex>
#include <nano/critical.h>
#include <nano/tuner/local.h>
#include <nano/tuner/surrogate.h>
#include <nano/tuner/util.h>

using namespace nano;

tuner_t::tuner_t(string_t id)
    : typed_t(std::move(id))
{
    register_parameter(parameter_t::make_integer("tuner::max_evals", 10, LE, 100, LE, 1000));
}

tuner_steps_t tuner_t::optimize(const param_spaces_t& spaces, const tuner_callback_t& callback,
                                const logger_t& logger) const
{
    critical(!spaces.empty(), "tuner: at least one parameter space is needed!");

    const auto max_evals = parameter("tuner::max_evals").value<size_t>();
    const auto min_igrid = make_min_igrid(spaces);
    const auto max_igrid = make_max_igrid(spaces);
    const auto avg_igrid = make_avg_igrid(spaces);

    tuner_steps_t steps;

    // initialize using a coarse grid
    evaluate(spaces, callback, igrids_t{avg_igrid}, logger, steps);
    for (tensor_size_t radius = 2; !steps.empty() && steps.size() < max_evals / 2; radius *= 2)
    {
        const auto igrids = local_search(min_igrid, max_igrid, steps.begin()->m_igrid, radius);
        if (!evaluate(spaces, callback, igrids, logger, steps))
        {
            break;
        }
    }

    do_optimize(spaces, callback, logger, steps);

    return steps;
}

factory_t<tuner_t>& tuner_t::all()
{
    static auto manager = factory_t<tuner_t>{};
    const auto  op      = []()
    {
        manager.add<local_search_tuner_t>("local search around the current optimum");
        manager.add<surrogate_tuner_t>("fit and minimize a quadratic surrogate function");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
