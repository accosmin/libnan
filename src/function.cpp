#include <mutex>

#include <function/benchmark/axis_ellipsoid.h>
#include <function/benchmark/cauchy.h>
#include <function/benchmark/chained_cb3I.h>
#include <function/benchmark/chained_cb3II.h>
#include <function/benchmark/chained_lq.h>
#include <function/benchmark/chung_reynolds.h>
#include <function/benchmark/dixon_price.h>
#include <function/benchmark/elastic_net.h>
#include <function/benchmark/exponential.h>
#include <function/benchmark/geometric.h>
#include <function/benchmark/kinks.h>
#include <function/benchmark/maxhilb.h>
#include <function/benchmark/maxq.h>
#include <function/benchmark/maxquad.h>
#include <function/benchmark/powell.h>
#include <function/benchmark/qing.h>
#include <function/benchmark/quadratic.h>
#include <function/benchmark/rosenbrock.h>
#include <function/benchmark/rotated_ellipsoid.h>
#include <function/benchmark/sargan.h>
#include <function/benchmark/schumer_steiglitz.h>
#include <function/benchmark/sphere.h>
#include <function/benchmark/styblinski_tang.h>
#include <function/benchmark/trid.h>
#include <function/benchmark/zakharov.h>
#include <nano/core/strutil.h>

using namespace nano;

function_t::function_t(string_t id, tensor_size_t size)
    : typed_t(std::move(id))
    , m_size(size)
{
}

void function_t::convex(const convexity c)
{
    m_convexity = c;
}

void function_t::smooth(const smoothness s)
{
    m_smoothness = s;
}

void function_t::strong_convexity(const scalar_t strong_convexity)
{
    m_strong_convexity = strong_convexity;
}

string_t function_t::name(const bool with_size) const
{
    return with_size ? scat(type_id(), "[", size(), "D]") : type_id();
}

bool function_t::constrain(constraint_t&& constraint)
{
    if (compatible(constraint, *this))
    {
        m_constraints.emplace_back(std::move(constraint));
        return true;
    }
    return false;
}

const constraints_t& function_t::constraints() const
{
    return m_constraints;
}

bool function_t::valid(const vector_t& x) const
{
    const auto op = [&](const constraint_t& constraint)
    { return ::nano::valid(constraint, x) < std::numeric_limits<scalar_t>::epsilon(); };

    return x.size() == size() && std::all_of(m_constraints.begin(), m_constraints.end(), op);
}

tensor_size_t function_t::n_equalities() const
{
    return ::nano::n_equalities(m_constraints);
}

tensor_size_t function_t::n_inequalities() const
{
    return ::nano::n_inequalities(m_constraints);
}

scalar_t function_t::operator()(vector_cmap_t x, vector_map_t gx) const
{
    assert(x.size() == size());
    assert(gx.size() == 0 || gx.size() == size());

    m_fcalls += 1;
    m_gcalls += (gx.size() == size()) ? 1 : 0;
    return do_vgrad(x, gx);
}

tensor_size_t function_t::fcalls() const
{
    return m_fcalls;
}

tensor_size_t function_t::gcalls() const
{
    return m_gcalls;
}

void function_t::clear_statistics() const
{
    m_fcalls = 0;
    m_gcalls = 0;
}

rfunction_t function_t::make(tensor_size_t, tensor_size_t) const
{
    return rfunction_t{};
}

bool function_t::optimum(vector_t xbest)
{
    if (xbest.size() != size())
    {
        return false;
    }
    else
    {
        m_optimum.m_status = optimum_t::status::solvable;
        m_optimum.m_xbest  = std::move(xbest);
        m_optimum.m_fbest  = do_vgrad(m_optimum.m_xbest, vector_map_t{});
        return true;
    }
}

bool function_t::optimum(const scalar_t fbest)
{
    if (!std::isfinite(fbest))
    {
        return false;
    }
    else
    {
        m_optimum.m_status = optimum_t::status::solvable;
        m_optimum.m_fbest  = fbest;
        return true;
    }
}

bool function_t::optimum(const optimum_t::status status)
{
    m_optimum.m_status = status;
    return true;
}

const optimum_t& function_t::optimum() const
{
    return m_optimum;
}

factory_t<function_t>& function_t::all()
{
    static auto manager = factory_t<function_t>{};
    const auto  op      = []()
    {
        manager.add<function_maxq_t>("MAXQ function: f(x) = max(i, x_i^2)");
        manager.add<function_maxquad_t>("MAXQUAD function: f(x) = max(k, x.dot(A_k*x) - b_k.dot(x))");
        manager.add<function_maxhilb_t>("MAXHILB function: f(x) = max(i, sum(j, xj / (i + j = 1))");
        manager.add<function_chained_lq_t>("chained LQ function (see documentation)");
        manager.add<function_chained_cb3I_t>("chained CB3 I function (see documentation)");
        manager.add<function_chained_cb3II_t>("chained CB3 II function (see documentation)");

        manager.add<function_trid_t>("Trid function: https://www.sfu.ca/~ssurjano/trid.html");
        manager.add<function_qing_t>("Qing function: http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html");
        manager.add<function_kinks_t>("random kinks: f(x) = sum(|x - K_i|, i)");
        manager.add<function_cauchy_t>("Cauchy function: f(x) = log(1 + x.dot(x))");
        manager.add<function_sargan_t>(
            "Sargan function: http://infinity77.net/global_optimization/test_functions_nd_S.html");
        manager.add<function_powell_t>("Powell function: https://www.sfu.ca/~ssurjano/powell.html");
        manager.add<function_sphere_t>("sphere function: f(x) = x.dot(x)");
        manager.add<function_zakharov_t>("Zakharov function: https://www.sfu.ca/~ssurjano/zakharov.html");
        manager.add<function_quadratic_t>("random quadratic function: f(x) = x.dot(a) + x * A * x, where A is PD");
        manager.add<function_rosenbrock_t>(
            "Rosenbrock function: https://en.wikipedia.org/wiki/Test_functions_for_optimization");
        manager.add<function_exponential_t>("exponential function: f(x) = exp(1 + x.dot(x) / D)");
        manager.add<function_dixon_price_t>("Dixon-Price function: https://www.sfu.ca/~ssurjano/dixonpr.html");
        manager.add<function_chung_reynolds_t>("Chung-Reynolds function: f(x) = (x.dot(x))^2");
        manager.add<function_axis_ellipsoid_t>("axis-parallel hyper-ellipsoid function: f(x) = sum(i*x+i^2, i=1,D)");
        manager.add<function_styblinski_tang_t>("Styblinski-Tang function: https://www.sfu.ca/~ssurjano/stybtang.html");
        manager.add<function_schumer_steiglitz_t>("Schumer-Steiglitz No. 02 function: f(x) = sum(x_i^4, i=1,D)");
        manager.add<function_rotated_ellipsoid_t>(
            "rotated hyper-ellipsoid function: https://www.sfu.ca/~ssurjano/rothyp.html");
        manager.add<function_geometric_optimization_t>(
            "generic geometric optimization function: f(x) = sum(i, exp(alpha_i + a_i.dot(x)))");

        manager.add<function_enet_mse_t>("mean squared error with ridge-like regularization", 10, 0.0, 1e+0);
        manager.add<function_enet_mse_t>("mean squared error with ridge-like regularization", 10, 0.0, 1e+2);
        manager.add<function_enet_mse_t>("mean squared error with ridge-like regularization", 10, 0.0, 1e+4);
        manager.add<function_enet_mse_t>("mean squared error with ridge-like regularization", 10, 0.0, 1e+6);

        manager.add<function_enet_mse_t>("mean squared error with lasso-like regularization", 10, 1e+0, 0.0);
        manager.add<function_enet_mse_t>("mean squared error with lasso-like regularization", 10, 1e+2, 0.0);
        manager.add<function_enet_mse_t>("mean squared error with lasso-like regularization", 10, 1e+4, 0.0);
        manager.add<function_enet_mse_t>("mean squared error with lasso-like regularization", 10, 1e+6, 0.0);

        manager.add<function_enet_mse_t>("mean squared error with elastic net-like regularization", 10, 1e+0, 1e+0);
        manager.add<function_enet_mse_t>("mean squared error with elastic net-like regularization", 10, 1e+2, 1e+2);
        manager.add<function_enet_mse_t>("mean squared error with elastic net-like regularization", 10, 1e+4, 1e+4);
        manager.add<function_enet_mse_t>("mean squared error with elastic net-like regularization", 10, 1e+6, 1e+6);

        manager.add<function_enet_mae_t>("mean absolute error with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_mae_t>("mean absolute error with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_mae_t>("mean absolute error with elastic net-like regularization", 10, 1.0, 1.0);

        manager.add<function_enet_hinge_t>("hinge loss (linear SVM) with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_hinge_t>("hinge loss (linear SVM) with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_hinge_t>("hinge loss (linear SVM) with elastic net-like regularization", 10, 1.0,
                                           1.0);

        manager.add<function_enet_cauchy_t>("cauchy loss (robust regression) with ridge-like regularization", 10, 0.0,
                                            1.0);
        manager.add<function_enet_cauchy_t>("cauchy loss (robust regression) with lasso-like regularization", 10, 1.0,
                                            0.0);
        manager.add<function_enet_cauchy_t>("cauchy loss (robust regression) with elastic net-like regularization", 10,
                                            1.0, 1.0);

        manager.add<function_enet_logistic_t>("logistic regression with ridge-like regularization", 10, 0.0, 1.0);
        manager.add<function_enet_logistic_t>("logistic regression with lasso-like regularization", 10, 1.0, 0.0);
        manager.add<function_enet_logistic_t>("logistic regression with elastic net-like regularization", 10, 1.0, 1.0);
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}

rfunctions_t function_t::make(const function_t::config_t& config, const std::regex& id_regex)
{
    const auto convexity  = config.m_convexity;
    const auto smoothness = config.m_smoothness;

    const auto min_dims = std::min(config.m_min_dims, config.m_max_dims);
    const auto max_dims = std::max(config.m_min_dims, config.m_max_dims);
    assert(min_dims >= 1);

    const auto& factory = function_t::all();
    const auto  ids     = factory.ids(id_regex);

    rfunctions_t functions;
    for (tensor_size_t dims = min_dims; dims <= max_dims;)
    {
        for (const auto& id : ids)
        {
            auto function = factory.get(id);

            if ((convexity == convexity::ignore || (function->convex() == (convexity == convexity::yes))) &&
                (smoothness == smoothness::ignore || (function->smooth() == (smoothness == smoothness::yes))))
            {
                functions.push_back(function->make(dims, config.m_summands));
            }
        }

        if (dims < 4)
        {
            ++dims;
        }
        else
        {
            dims *= 2;
        }
    }

    return functions;
} // LCOV_EXCL_LINE
