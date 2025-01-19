#include <nano/core/combinatorial.h>
#include <nano/critical.h>
#include <nano/logger.h>
#include <nano/tuner/util.h>

using namespace nano;

igrid_t nano::make_min_igrid(const param_spaces_t& spaces)
{
    const auto n_params = static_cast<tensor_size_t>(spaces.size());

    return make_full_tensor<tensor_size_t>(make_dims(n_params), 0);
}

igrid_t nano::make_max_igrid(const param_spaces_t& spaces)
{
    const auto n_params = static_cast<tensor_size_t>(spaces.size());

    auto igrid = igrid_t{n_params};
    for (tensor_size_t iparam = 0; iparam < n_params; ++iparam)
    {
        const auto& space = spaces[static_cast<size_t>(iparam)];
        igrid(iparam)     = space.values().size() - 1;
    }
    return igrid;
}

igrid_t nano::make_avg_igrid(const param_spaces_t& spaces)
{
    const auto n_params = static_cast<tensor_size_t>(spaces.size());

    auto igrid = igrid_t{n_params};
    for (tensor_size_t iparam = 0; iparam < n_params; ++iparam)
    {
        const auto& space = spaces[static_cast<size_t>(iparam)];
        igrid(iparam)     = space.values().size() / 2;
    }
    return igrid;
}

tensor2d_t nano::map_to_grid(const param_spaces_t& spaces, const std::vector<igrid_t>& igrids)
{
    const auto n_params = static_cast<tensor_size_t>(spaces.size());
    const auto n_trials = static_cast<tensor_size_t>(igrids.size());

    auto values = tensor2d_t{n_trials, n_params};
    for (tensor_size_t itrial = 0; itrial < n_trials; ++itrial)
    {
        const auto& igrid = igrids[static_cast<size_t>(itrial)];
        for (tensor_size_t iparam = 0; iparam < n_params; ++iparam)
        {
            const auto& space      = spaces[static_cast<size_t>(iparam)];
            values(itrial, iparam) = space.values()(igrid(iparam));
        }
    }
    return values;
}

igrids_t nano::local_search(const igrid_t& min_igrid, const igrid_t& max_igrid, const igrid_t& src_igrid,
                            const tensor_size_t radius)
{
    const auto n_params         = min_igrid.size();
    const auto trials_per_space = make_full_tensor<tensor_size_t>(make_dims(n_params), 3);

    igrids_t igrids;
    for (auto it = combinatorial_iterator_t{trials_per_space}; it; ++it)
    {
        auto igrid    = *it;
        igrid.array() = (igrid.array() - 1) * radius + src_igrid.array();

        // check that the current trial point is not outside the grid
        if ((igrid.array() - min_igrid.array()).minCoeff() < 0 || (max_igrid.array() - igrid.array()).minCoeff() < 0)
        {
            continue;
        }

        igrids.emplace_back(std::move(igrid));
    }

    return igrids;
}

bool nano::evaluate(const param_spaces_t& spaces, const tuner_callback_t& callback, igrids_t igrids, const logger_t&,
                    tuner_steps_t& steps)
{
    // no need to consider grid points already evaluated
    const auto op = [&](const igrid_t& igrid)
    {
        const auto _ = [&](const auto& step) { return step.m_igrid == igrid; };
        return std::find_if(steps.begin(), steps.end(), _) != steps.end();
    };
    const auto it = std::remove_if(igrids.begin(), igrids.end(), op);
    igrids.erase(it, igrids.end());

    if (igrids.empty())
    {
        return false;
    }

    // evaluate the new grid points...
    const auto before = steps.size();
    const auto params = map_to_grid(spaces, igrids);
    const auto values = callback(params);

    for (tensor_size_t itrial = 0; itrial < values.size(); ++itrial)
    {
        const auto& igrid = igrids[static_cast<size_t>(itrial)];

        critical(std::isfinite(values(itrial)), "tuner: invalid value (", values(itrial), ") detected for parameters (",
                 params.vector(itrial).transpose(), ")!");

        steps.emplace_back(tuner_step_t{igrid, params.tensor(itrial), values(itrial)});
    }

    // NB: the evaluated steps are always sorted so that the first one is the optimum!
    std::sort(steps.begin(), steps.end());

    return steps.size() != before;
}
