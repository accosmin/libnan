#include <fixture/function.h>
#include <fixture/loss.h>
#include <nano/core/numeric.h>
#include <nano/core/random.h>
#include <nano/function.h>
#include <nano/function/util.h>
#include <nano/loss.h>
#include <nano/loss/class.h>
#include <utest/utest.h>

using namespace nano;

struct loss_function_t final : public function_t
{
    loss_function_t(const rloss_t& loss, const tensor_size_t xmaps)
        : function_t("loss", 3 * xmaps)
        , m_loss(loss)
        , m_target(3, xmaps, 1, 1)
        , m_values(3)
    {
        m_target.tensor(0) = class_target(xmaps, 11 % xmaps);
        m_target.tensor(1) = class_target(xmaps, 12 % xmaps);
        m_target.tensor(2) = class_target(xmaps, 13 % xmaps);

        convex(m_loss->convex() ? convexity::yes : convexity::no);
        smooth(m_loss->smooth() ? smoothness::yes : smoothness::no);
    }

    rfunction_t clone() const override { return std::make_unique<loss_function_t>(*this); }

    scalar_t do_vgrad(vector_cmap_t x, vector_map_t gx) const override
    {
        UTEST_REQUIRE_EQUAL(x.size(), m_target.size());
        const auto output = map_tensor(x.data(), m_target.dims());

        if (gx.size() == x.size())
        {
            m_loss->vgrad(m_target, output, map_tensor(gx.data(), m_target.dims()));
            UTEST_REQUIRE(gx.array().isFinite().all());
        }

        m_loss->value(m_target, output, m_values.tensor());
        UTEST_REQUIRE(m_values.array().isFinite().all());
        return m_values.array().sum();
    }

    const rloss_t&     m_loss;
    tensor4d_t         m_target;
    mutable tensor1d_t m_values;
};

UTEST_BEGIN_MODULE(test_loss)

UTEST_CASE(factory)
{
    const auto& losses = loss_t::all();
    UTEST_CHECK_EQUAL(losses.ids().size(), 17U);
    UTEST_CHECK(losses.get("mae") != nullptr);
    UTEST_CHECK(losses.get("mse") != nullptr);
    UTEST_CHECK(losses.get("cauchy") != nullptr);
    UTEST_CHECK(losses.get("pinball") != nullptr);
    UTEST_CHECK(losses.get("m-hinge") != nullptr);
    UTEST_CHECK(losses.get("s-hinge") != nullptr);
    UTEST_CHECK(losses.get("s-classnll") != nullptr);
    UTEST_CHECK(losses.get("m-savage") != nullptr);
    UTEST_CHECK(losses.get("s-savage") != nullptr);
    UTEST_CHECK(losses.get("m-tangent") != nullptr);
    UTEST_CHECK(losses.get("s-tangent") != nullptr);
    UTEST_CHECK(losses.get("m-logistic") != nullptr);
    UTEST_CHECK(losses.get("s-logistic") != nullptr);
    UTEST_CHECK(losses.get("m-exponential") != nullptr);
    UTEST_CHECK(losses.get("s-exponential") != nullptr);
    UTEST_CHECK(losses.get("m-squared-hinge") != nullptr);
    UTEST_CHECK(losses.get("s-squared-hinge") != nullptr);
}

UTEST_CASE(gradient)
{
    const tensor_size_t cmd_min_dims = 2;
    const tensor_size_t cmd_max_dims = 5;

    for (const auto& loss_id : loss_t::all().ids())
    {
        UTEST_NAMED_CASE(loss_id);

        for (tensor_size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; ++cmd_dims)
        {
            const auto loss     = make_loss(loss_id);
            const auto function = loss_function_t(loss, cmd_dims);

            UTEST_CHECK_EQUAL(loss->convex(), function.convex());
            UTEST_CHECK_EQUAL(loss->smooth(), function.smooth());

            if (loss_id == "pinball")
            {
                loss->parameter("loss::pinball::alpha") = static_cast<scalar_t>(cmd_dims - cmd_min_dims + 1) /
                                                          static_cast<scalar_t>(cmd_max_dims - cmd_min_dims + 2);
            }

            const auto max_power = (loss_id == "m-exponential" || loss_id == "s-exponential") ? 5 : 20;
            for (int power = 0; power <= max_power; ++power)
            {
                auto tx = make_random_tensor<scalar_t>(make_dims(function.size()), -1.0, +1.0);
                tx.array() *= std::pow(std::exp(1.0), power);

                const vector_t x = tx.vector();

                const auto f = function(x);
                UTEST_CHECK_EQUAL(std::isfinite(f), true);
                UTEST_CHECK_GREATER_EQUAL(f, scalar_t(0));
                UTEST_CHECK_LESS(grad_accuracy(function, x), 5 * epsilon2<scalar_t>());
            }

            check_gradient(function);
            check_convexity(function);
        }
    }
}

UTEST_CASE(single_class)
{
    for (const auto& loss_id : loss_t::all().ids(std::regex("s-.+")))
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss = make_loss(loss_id);

        const auto n_classes = 1;

        tensor4d_t targets(4, n_classes, 1, 1);
        targets.tensor(0) = class_target(n_classes);
        targets.tensor(1) = class_target(n_classes, 0);
        targets.tensor(2) = class_target(n_classes);
        targets.tensor(3) = class_target(n_classes, 0);

        tensor4d_t outputs(4, n_classes, 1, 1);
        outputs.tensor(0) = class_target(n_classes);
        outputs.tensor(1) = class_target(n_classes, 0);
        outputs.tensor(2) = class_target(n_classes, 0);
        outputs.tensor(3) = class_target(n_classes);

        tensor1d_t errors;
        loss->error(targets, outputs, errors);

        UTEST_CHECK_CLOSE(errors(0), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(1), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(2), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(3), scalar_t(1), epsilon0<scalar_t>());
    }
}

UTEST_CASE(single_label_multi_class)
{
    for (const auto& loss_id : loss_t::all().ids(std::regex("s-.+")))
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss = make_loss(loss_id);

        const auto n_classes = 13;

        tensor4d_t targets(4, n_classes, 1, 1);
        targets.tensor(0) = class_target(n_classes, 11);
        targets.tensor(1) = class_target(n_classes, 11);
        targets.tensor(2) = class_target(n_classes, 11);
        targets.tensor(3) = class_target(n_classes, 11);

        tensor4d_t outputs(4, n_classes, 1, 1);
        outputs.tensor(0)    = class_target(n_classes, 11);
        outputs.tensor(1)    = class_target(n_classes, 12);
        outputs.tensor(2)    = class_target(n_classes, 11);
        outputs.vector(2)(7) = pos_target() + 1;
        outputs.tensor(3)    = class_target(n_classes);

        tensor1d_t errors;
        loss->error(targets, outputs, errors);

        UTEST_CHECK_CLOSE(errors(0), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(1), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(2), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(3), scalar_t(1), epsilon0<scalar_t>());
    }
}

UTEST_CASE(multi_label_multi_class)
{
    for (const auto& loss_id : loss_t::all().ids(std::regex("m-.+")))
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss = make_loss(loss_id);

        const auto n_classes = 13;

        tensor4d_t targets(6, n_classes, 1, 1);
        targets.tensor(0) = class_target(n_classes, 7, 9);
        targets.tensor(1) = class_target(n_classes, 7, 9);
        targets.tensor(2) = class_target(n_classes, 7, 9);
        targets.tensor(3) = class_target(n_classes, 7, 9);
        targets.tensor(4) = class_target(n_classes, 7, 9);
        targets.tensor(5) = class_target(n_classes, 7, 9);

        tensor4d_t outputs(6, n_classes, 1, 1);
        outputs.tensor(0) = class_target(n_classes, 7, 9);
        outputs.tensor(1) = class_target(n_classes);
        outputs.tensor(2) = class_target(n_classes, 5);
        outputs.tensor(3) = class_target(n_classes, 7);
        outputs.tensor(4) = class_target(n_classes, 5, 9);
        outputs.tensor(5) = class_target(n_classes, 7, 9, 11);

        tensor1d_t errors;
        loss->error(targets, outputs, errors);

        UTEST_CHECK_CLOSE(errors(0), scalar_t(0), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(1), scalar_t(2), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(2), scalar_t(3), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(3), scalar_t(1), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(4), scalar_t(2), epsilon0<scalar_t>());
        UTEST_CHECK_CLOSE(errors(5), scalar_t(1), epsilon0<scalar_t>());
    }
}

UTEST_CASE(regression)
{
    for (const auto& loss_id : {"mae", "mse", "cauchy"})
    {
        UTEST_NAMED_CASE(loss_id);

        const auto loss = make_loss(loss_id);

        const auto target = make_random_tensor<scalar_t>(make_dims(3, 4, 1, 1));

        tensor4d_t output = target;

        tensor1d_t errors(3);
        loss->error(target, output, errors.tensor());
        UTEST_CHECK_LESS(errors.array().abs().maxCoeff(), epsilon0<scalar_t>());
    }
}

UTEST_CASE(quantile_regression)
{
    const auto loss = make_loss("pinball");

    const auto targets = make_tensor<scalar_t>(make_dims(5, 2, 1, 1), 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
    const auto outputs = make_tensor<scalar_t>(make_dims(5, 2, 1, 1), 0.1, 0.0, 0.2, 0.4, 0.2, 0.3, 0.7, 0.9, 0.9, 0.6);

    auto values = tensor1d_t{5};
    auto errors = tensor1d_t{5};
    {
        loss->parameter("loss::pinball::alpha") = 0.5;
        loss->value(targets, outputs, values);
        loss->error(targets, outputs, errors);

        const auto expected = make_tensor<scalar_t>(make_dims(5), 0.10, 0.05, 0.20, 0.15, 0.20);
        UTEST_CHECK_CLOSE(values, expected, 1e-12);
        UTEST_CHECK_CLOSE(errors, expected, 1e-12);
    }
    {
        loss->parameter("loss::pinball::alpha") = 0.2;
        loss->value(targets, outputs, values);
        loss->error(targets, outputs, errors);

        const auto expected = make_tensor<scalar_t>(make_dims(5), 0.10, 0.08, 0.08, 0.24, 0.14);
        UTEST_CHECK_CLOSE(values, expected, 1e-12);
        UTEST_CHECK_CLOSE(errors, expected, 1e-12);
    }
    {
        loss->parameter("loss::pinball::alpha") = 0.8;
        loss->value(targets, outputs, values);
        loss->error(targets, outputs, errors);

        const auto expected = make_tensor<scalar_t>(make_dims(5), 0.10, 0.02, 0.32, 0.06, 0.26);
        UTEST_CHECK_CLOSE(values, expected, 1e-12);
        UTEST_CHECK_CLOSE(errors, expected, 1e-12);
    }
}

UTEST_END_MODULE()
