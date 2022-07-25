#pragma once

#include <nano/function/benchmark.h>

namespace nano
{
    ///
    /// \brief sphere function: f(x) = x.dot(x).
    ///
    class NANO_PUBLIC function_sphere_t final : public benchmark_function_t
    {
    public:
        ///
        /// \brief constructor
        ///
        explicit function_sphere_t(tensor_size_t dims = 10);

        ///
        /// \brief @see function_t
        ///
        scalar_t do_vgrad(const vector_t& x, vector_t* gx) const override;

        ///
        /// \brief @see benchmark_function_t
        ///
        rfunction_t make(tensor_size_t dims, tensor_size_t summands) const override;
    };
} // namespace nano