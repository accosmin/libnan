#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
namespace detail
{
template <class ttensor, class... ttensors>
auto size(const ttensor& tensor, const ttensors&...)
{
    return tensor.template size<0>();
}

template <class ttensor>
void copy(const tensor_size_t isrc, const tensor_size_t idst, ttensor& tensor)
{
    assert(isrc < tensor.template size<0>());
    assert(idst < tensor.template size<0>());

    if constexpr (ttensor::rank() == 1U)
    {
        tensor(idst) = tensor(isrc);
    }
    else
    {
        tensor.tensor(idst) = tensor.tensor(isrc);
    }
}
} // namespace detail

///
/// \brief remove all sub-tensors indexed by the first dimension flagged by the given operator
///     and return the number of sub-tensors that are kept.
///
/// NB: the remaining ones are compacted starting the begining and their number is returned.
/// NB: no allocation is performed.
///
template <class toperator, class... ttensors>
auto remove_if(const toperator& op, ttensors&&... tensors) ->
    typename std::enable_if_t<(is_tensor_v<std::remove_reference_t<ttensors>> && ...), tensor_size_t>
{
    const auto last = detail::size(std::forward<ttensors>(tensors)...);

    auto first = tensor_size_t{0};
    for (; first != last && !op(first); ++first)
    {
    }

    if (first != last)
    {
        for (tensor_size_t i = first; ++i != last;)
        {
            if (!op(i))
            {
                (detail::copy(i, first, tensors), ...);
                ++first;
            }
        }
    }

    return first;
}
} // namespace nano
