#include <nano/core/cmdline.h>
#include <nano/main.h>
#include <nano/solver/augmented.h>
#include <nano/solver/penalty.h>

using namespace nano;

namespace
{
// TODO: generate benchmark linear and quadratic programs
// TODO: associate names to programs expected_t{[linear|quadratic][nD][param=value]}

struct result_t
{
    result_t() = default;

    result_t(const solver_state_t& state, const int64_t milliseconds)
        : m_value(state.fx())
        , m_status(state.status())
        , m_fcalls(state.fcalls())
        , m_gcalls(state.gcalls())
        , m_milliseconds(milliseconds)
    {
    }

    scalar_t      m_value{0.0};
    scalar_t      m_kkt{0.0};
    solver_status m_status{solver_status::converged};
    tensor_size_t m_fcalls{0};
    tensor_size_t m_gcalls{0};
    int64_t       m_milliseconds{0};
};

/*struct solver_stats_t
{
    explicit solver_stats_t(size_t trials)
        : m_values(static_cast<tensor_size_t>(trials))
        , m_gnorms(static_cast<tensor_size_t>(trials))
        , m_errors(static_cast<tensor_size_t>(trials))
        , m_maxits(static_cast<tensor_size_t>(trials))
        , m_fcalls(static_cast<tensor_size_t>(trials))
        , m_gcalls(static_cast<tensor_size_t>(trials))
        , m_millis(static_cast<tensor_size_t>(trials))
        , m_ranks(static_cast<tensor_size_t>(trials))
        , m_precisions(static_cast<tensor_size_t>(trials))
    {
    }

    void update(size_t trial, const result_t& result, scalar_t precision, ptrdiff_t rank)
    {
        const auto itrial    = static_cast<tensor_size_t>(trial);
        m_values(itrial)     = result.m_value;
        m_gnorms(itrial)     = result.m_gnorm;
        m_errors(itrial)     = (result.m_status == solver_status::failed) ? 1.0 : 0.0;
        m_maxits(itrial)     = (result.m_status == solver_status::max_iters) ? 1.0 : 0.0;
        m_fcalls(itrial)     = static_cast<scalar_t>(result.m_fcalls);
        m_gcalls(itrial)     = static_cast<scalar_t>(result.m_gcalls);
        m_millis(itrial)     = static_cast<scalar_t>(result.m_milliseconds);
        m_ranks(itrial)      = static_cast<scalar_t>(rank);
        m_precisions(itrial) = precision;
    }

    void update(size_t trial, const solver_stats_t& stats)
    {
        const auto itrial    = static_cast<tensor_size_t>(trial);
        m_values(itrial)     = std::numeric_limits<scalar_t>::quiet_NaN();
        m_gnorms(itrial)     = stats.m_gnorms.mean();
        m_errors(itrial)     = stats.m_errors.sum();
        m_maxits(itrial)     = stats.m_maxits.sum();
        m_fcalls(itrial)     = stats.m_fcalls.mean();
        m_gcalls(itrial)     = stats.m_gcalls.mean();
        m_millis(itrial)     = stats.m_millis.mean();
        m_ranks(itrial)      = stats.m_ranks.mean();
        m_precisions(itrial) = stats.m_precisions.mean();
    }

    // attributes
    tensor1d_t m_values;     ///< function values
    tensor1d_t m_gnorms;     ///< gradient norms
    tensor1d_t m_errors;     ///< #internal errors (e.g. line-search failed)
    tensor1d_t m_maxits;     ///< #maximum iterations reached (without convergence)
    tensor1d_t m_fcalls;     ///< #function value calls
    tensor1d_t m_gcalls;     ///< #gradient calls
    tensor1d_t m_millis;     ///< number of milliseconds
    tensor1d_t m_ranks;      ///< rank as ordered by the function value
    tensor1d_t m_precisions; ///< relative precision to the best solver
};

auto relative_precision(const scalar_t value, const scalar_t best_value, const scalar_t epsilon)
{
    assert(value >= best_value);
    return std::log10(std::max(value - best_value, epsilon));
}

auto relative_precision(const result_t& result, const result_t& best_result, const scalar_t epsilon)
{
    return relative_precision(result.m_value, best_result.m_value, epsilon);
}

auto make_solver_name(const rsolver_t& solver)
{
    return solver->type() == solver_type::line_search
             ? scat(solver->type_id(), " [", solver->lsearch0().type_id(), ",", solver->lsearchk().type_id(), "]")
             : solver->type_id();
}

using points_t = std::vector<vector_t>;
using results_t = std::vector<result_t>;
using solvers_t = std::vector<rsolver_t>;

auto& print_scalar(row_t& row, const scalar_t value)
{
    return std::isfinite(value) ? (row << value) : (row << "N/A");
}

auto& print_integer(row_t& row, const scalar_t value)
{
    return std::isfinite(value) ? (row << static_cast<size_t>(value)) : (row << "N/A");
}

void print_table(const string_t& table_name, const solvers_t& solvers, const std::vector<solver_stats_t>& stats)
{
    // gather statistics per solver
    const auto max_evals        = solvers[0U]->parameter("solver::max_evals").value<int>();
    const auto max_digits_calls = static_cast<size_t>(std::log10(max_evals)) + 1U;

    // display per-function statistics
    table_t table;
    table.header() << align(table_name, 32) << align("precision", 9) << align("rank", 4) << align("value", 12)
                   << align("gnorm", 12) << "errors"
                   << "maxits" << align("fcalls", max_digits_calls) << align("gcalls", max_digits_calls)
                   << align("[ms]", 5);
    table.delim();

    for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
    {
        const auto& stat        = stats[isolver];
        const auto  solver_name = make_solver_name(solvers[isolver]);

        auto& row = table.append();
        row << solver_name << scat(std::fixed, std::setprecision(4), stat.m_precisions.mean())
            << scat(std::fixed, std::setprecision(2), stat.m_ranks.mean());
        print_scalar(row, stat.m_values.mean());
        print_scalar(row, stat.m_gnorms.mean());
        print_integer(row, stat.m_errors.sum());
        print_integer(row, stat.m_maxits.sum());
        print_integer(row, stat.m_fcalls.mean());
        print_integer(row, stat.m_gcalls.mean());
        print_integer(row, stat.m_millis.mean());
    }

    assert(table.rows() == solvers.size() + 2U);

    table.sort(nano::make_less_from_string<scalar_t>(), {1}); // NB: sort solvers by the average precision!
    std::cout << table;
}

auto minimize_all(parallel::pool_t& pool, const function_t& function, const solvers_t& solvers, const points_t& x0s,
                  const string_t& log_dir)
{
    const auto minimize_one = [&](const size_t i)
    {
        const auto itrial  = i / solvers.size();
        const auto isolver = i % solvers.size();

        const auto& x0     = x0s[itrial];
        const auto  solver = solvers[isolver]->clone();

        const auto logger = [&]()
        {
            if (!log_dir.empty())
            {
                const auto path = std::filesystem::path(log_dir) / function.name() / scat("trial", itrial + 1) /
                                  scat(solver->type_id(), ".log");

                return make_file_logger(path.string());
            }
            else
            {
                return make_null_logger();
            }
        }();

        const auto timer = nano::timer_t{};
        const auto state = solver->minimize(*function.clone(), x0, logger);
        const auto milli = timer.milliseconds().count();

        return result_t{state, milli};
    };

    auto results = results_t{x0s.size() * solvers.size()};
    pool.map(results.size(), [&](const size_t i, const size_t) { results[i] = minimize_one(i); });
    return results;
}

auto benchmark(parallel::pool_t& pool, const function_t& function, const solvers_t& solvers, const size_t trials,
               const string_t& log_dir)
{
    // generate a fixed set of random initial points
    points_t x0s(trials);
    std::generate(x0s.begin(), x0s.end(), [&]() { return make_random_vector<scalar_t>(function.size()); });

    // and minimize in parallel all (solver, random initial point) combinations
    const auto results = minimize_all(pool, function, solvers, x0s, log_dir);

    // gather statistics per solver
    auto stats = std::vector<solver_stats_t>{solvers.size(), solver_stats_t{trials}};
    auto ranks = std::vector<std::pair<scalar_t, size_t>>{solvers.size()};

    for (size_t trial = 0U; trial < trials; ++trial)
    {
        const auto* const begin = &results[trial * solvers.size()];
        const auto* const end   = &results[trial * solvers.size() + solvers.size()];

        const auto& best_result =
            *std::min_element(begin, end, [](const auto& lhs, const auto& rhs) { return lhs.m_value < rhs.m_value; });

        for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
        {
            const auto& result = results[trial * solvers.size() + isolver];
            ranks[isolver]     = std::make_pair(result.m_value, isolver);
        }
        std::sort(ranks.begin(), ranks.end());

        for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
        {
            const auto& result = results[trial * solvers.size() + isolver];
            assert(std::isfinite(result.m_value));
            assert(std::isfinite(result.m_gnorm));

            const auto epsilon   = solvers[isolver]->parameter("solver::epsilon").value<scalar_t>();
            const auto precision = relative_precision(result, best_result, epsilon);

            const auto find = [&](const auto& v) { return v.second == isolver; };
            const auto rank = (std::find_if(ranks.begin(), ranks.end(), find) - ranks.begin()) + 1;

            stats[isolver].update(trial, result, precision, rank);
        }
    }

    // display per-function statistics
    print_table(function.name(), solvers, stats);

    return stats;
}*/

int unsafe_main(int argc, const char* argv[])
{
    using namespace nano;

    // parse the command line
    cmdline_t cmdline("benchmark solvers");
    cmdline.add("--min-dims", "minimum number of dimensions for each test function (if feasible)", "4");
    cmdline.add("--max-dims", "maximum number of dimensions for each test function (if feasible)", "16");
    cmdline.add("--trials", "number of random trials for each test function", "100");
    cmdline.add("--log-dir", "directory to log the optimization trajectories");

    const auto options = cmdline.process(argc, argv);
    if (cmdline.handle(options))
    {
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto min_dims = options.get<tensor_size_t>("--min-dims");
    const auto max_dims = options.get<tensor_size_t>("--max-dims");
    const auto trials   = options.get<size_t>("--trials");

    const auto log_dir = options.has("--log-dir") ? options.get("--log-dir") : string_t{};

    (void)min_dims;
    (void)max_dims;
    (void)trials;
    (void)log_dir;

    /*
    const auto functions = function_t::make({min_dims, max_dims, convex, smooth}, fregex);
    critical(functions.empty(), "at least a function needs to be selected!");

    auto rconfig = cmdconfig_t{options};

    // construct the list of solver configurations to evaluate
    solvers_t solvers;
    for (const auto& solver_id : solver_ids)
    {
        auto solver = solver_t::all().get(solver_id);
        if (solver->type() == solver_type::line_search)
        {
            for (const auto& lsearch0_id : lsearch0_ids)
            {
                for (const auto& lsearchk_id : lsearchk_ids)
                {
                    solver        = solver_t::all().get(solver_id);
                    auto lsearch0 = lsearch0_t::all().get(lsearch0_id);
                    auto lsearchk = lsearchk_t::all().get(lsearchk_id);

                    rconfig.setup(*solver);
                    rconfig.setup(*lsearch0);
                    rconfig.setup(*lsearchk);

                    solver->lsearch0(*lsearch0);
                    solver->lsearchk(*lsearchk);

                    solvers.emplace_back(std::move(solver));
                }
            }
        }
        else
        {
            rconfig.setup(*solver);
            solvers.emplace_back(std::move(solver));
        }
    }

    // benchmark solvers and display statistics independently per function
    auto thread_pool  = parallel::pool_t{};
    auto solver_stats = std::vector<solver_stats_t>{solvers.size(), solver_stats_t{functions.size()}};

    for (size_t ifunction = 0U; ifunction < functions.size(); ++ifunction)
    {
        const auto& function = functions[ifunction];
        const auto  funstats = benchmark(thread_pool, *function, solvers, trials, log_dir);
        for (size_t isolver = 0U; isolver < solvers.size(); ++isolver)
        {
            const auto& stats = funstats[isolver];
            solver_stats[isolver].update(ifunction, stats);
        }
    }

    // display global statistics
    print_table("solver", solvers, solver_stats);
    */

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
