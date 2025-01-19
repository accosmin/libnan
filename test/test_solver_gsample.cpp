#include <fixture/solver.h>

using namespace nano;

namespace
{
auto make_solver_ids()
{
    return strings_t{"ellipsoid", "gs", "gs-lbfgs", "ags", "ags-lbfgs"};
}
} // namespace

UTEST_BEGIN_MODULE(test_solver_gsample)

UTEST_CASE(smooth)
{
    check_minimize(make_solver_ids(), function_t::make({4, 4, convexity::yes, smoothness::yes, 100}));
}

UTEST_CASE(nonsmooth)
{
    check_minimize(make_solver_ids(), function_t::make({4, 4, convexity::yes, smoothness::no, 100}));
}

UTEST_END_MODULE()
