#include <iomanip>
#include <nano/core/chrono.h>
#include <nano/core/cmdline.h>
#include <nano/core/factory_util.h>
#include <nano/core/logger.h>
#include <nano/core/stats.h>
#include <nano/core/table.h>
#include <nano/function/util.h>

using namespace nano;

namespace
{
void eval_func(const function_t& function, table_t& table)
{
    const auto dims = function.size();
    const vector_t x = vector_t::Zero(dims);
    vector_t g = vector_t::Zero(dims);

    const size_t trials = 16;

    volatile scalar_t fx = 0;
    volatile scalar_t gx = 0;

    const auto measure_fval = [&]()
    {
        const auto old_value = fx;

        fx = old_value + function.vgrad(x);
    };
    const auto measure_grad = [&]()
    {
        const auto old_value = gx;

        function.vgrad(x, &g);
        gx = old_value + g.template lpNorm<Eigen::Infinity>();
    };

    const auto fval_time = measure<nanoseconds_t>(measure_fval, trials).count();
    const auto grad_time = measure<nanoseconds_t>(measure_grad, trials).count();

    scalar_t grad_accuracy = 0;
    for (size_t i = 0; i < trials; ++ i)
    {
        grad_accuracy += ::nano::grad_accuracy(function, vector_t::Random(dims));
    }

    auto& row = table.append();
    row << function.name() << fval_time << grad_time
        << scat(std::setprecision(12), std::fixed, grad_accuracy / static_cast<scalar_t>(trials));
}

int unsafe_main(int argc, const char* argv[])
{
    // parse the command line
    cmdline_t cmdline("benchmark optimization test functions");
    cmdline.add("", "min-dims",         "minimum number of dimensions for each test function (if feasible)", "1024");
    cmdline.add("", "max-dims",         "maximum number of dimensions for each test function (if feasible)", "1024");
    cmdline.add("", "function",         "use this regex to select the functions to benchmark", ".+");
    cmdline.add("", "list-function",    "list the available test functions");

    const auto options = cmdline.process(argc, argv);

    if (options.has("help"))
    {
        cmdline.usage();
        return EXIT_SUCCESS;
    }

    if (options.has("list-function"))
    {
        std::cout << make_table("function", function_t::all(), options.get<string_t>("function"));
        return EXIT_SUCCESS;
    }

    // check arguments and options
    const auto min_dims = options.get<tensor_size_t>("min-dims");
    const auto max_dims = options.get<tensor_size_t>("max-dims");
    const auto fregex = std::regex(options.get<string_t>("function"));
    const auto fconfig = function_t::config_t{min_dims, max_dims, convexity::ignore, smoothness::ignore};

    table_t table;
    table.header() << "function" << "f(x)[ns]" << "f(x,g)[ns]" << "grad accuracy";
    table.delim();

    tensor_size_t prev_size = min_dims;
    for (const auto& function : function_t::make(fconfig, fregex))
    {
        if (function->size() != prev_size)
        {
            table.delim();
            prev_size = function->size();
        }
        eval_func(*function, table);
    }

    std::cout << table;

    // OK
    return EXIT_SUCCESS;
}
} // namespace

int main(int argc, const char* argv[])
{
    return nano::safe_main(unsafe_main, argc, argv);
}
