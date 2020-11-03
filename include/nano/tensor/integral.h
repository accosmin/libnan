#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
    ///
    /// \brief compute the integral of a tensor of arbitrary rank (aka the sum-area table).
    ///
    template <size_t trank>
    struct integral_t
    {
        template
        <
            template <typename, size_t> class tstoragei, typename tscalari,
            template <typename, size_t> class tstorageo, typename tscalaro
        >
        static void get(
            const tensor_t<tstoragei, tscalari, trank>& itensor, tensor_t<tstorageo, tscalaro, trank>&& otensor)
        {
            for (tensor_size_t i0 = 0, size0 = itensor.template size<0>(); i0 < size0; ++ i0)
            {
                integral_t<trank - 1>::get(itensor.tensor(i0), otensor.tensor(i0));
                if (i0 > 0)
                {
                    otensor.vector(i0) += otensor.vector(i0 - 1);
                }
            }
        }
    };

    template <>
    struct integral_t<1>
    {
        template
        <
            template <typename, size_t> class tstoragei, typename tscalari,
            template <typename, size_t> class tstorageo, typename tscalaro
        >
        static void get(
            const tensor_t<tstoragei, tscalari, 1>& itensor, tensor_t<tstorageo, tscalaro, 1>&& otensor)
        {
            otensor(0) = itensor(0);
            for (tensor_size_t i0 = 1, size0 = itensor.template size<0>(); i0 < size0; ++ i0)
            {
                otensor(i0) = otensor(i0 - 1) + itensor(i0);
            }
        }
    };

    template
    <
        size_t trank,
        template <typename, size_t> class tstoragei, typename tscalari,
        template <typename, size_t> class tstorageo, typename tscalaro
    >
    void integral(const tensor_t<tstoragei, tscalari, trank>& itensor, tensor_t<tstorageo, tscalaro, trank>& otensor)
    {
        assert(itensor.dims() == otensor.dims());
        if (itensor.size() > 0)
        {
            integral_t<trank>::get(itensor, std::move(otensor));
        }
    }
}
