#pragma once

#include <nano/function.h>

namespace nano
{
    ///
    /// \brief maps a given lambda to the function_t interface.
    ///
    template <typename tlambda>
    class NANO_PUBLIC lambda_function_t final : public function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit lambda_function_t(const tensor_size_t dims, const convexity convex, const smoothness smooth,
                                   const scalar_t strong_convexity, tlambda lambda)
            : function_t(typeid(tlambda).name(), dims)
            , m_lambda(std::move(lambda))
        {
            function_t::convex(convex);
            function_t::smooth(smooth);
            function_t::strong_convexity(strong_convexity);
        }

        ///
        /// \brief @see clonable_t
        ///
        rfunction_t clone() const override { return std::make_unique<lambda_function_t>(*this); }

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override
        {
            assert(x.size() == size());
            assert(gx == nullptr || gx->size() == size());

            return m_lambda(x, gx);
        }

    private:
        // attributes
        tlambda m_lambda{}; ///<
    };

    ///
    /// \brief create a compatible function_t from the given lambda.
    ///
    template <typename tlambda>
    auto make_function(const tensor_size_t dims, const convexity convex, const smoothness smooth,
                       const scalar_t strong_convexity, tlambda lambda)
    {
        return lambda_function_t{dims, convex, smooth, strong_convexity, std::move(lambda)};
    }
} // namespace nano
