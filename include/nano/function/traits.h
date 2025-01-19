#pragma once

#include <nano/tensor/tensor.h>

namespace nano
{
template <class T>
struct is_tensor_matrix : std::false_type
{
};

template <template <class, size_t> class tstorage, class tscalar, size_t trank>
struct is_tensor_matrix<tensor_t<tstorage, tscalar, trank>>
    : std::conditional<trank <= 2U, std::true_type, std::false_type>::type
{
};

template <class T>
struct is_tensor_vector : std::false_type
{
};

template <template <class, size_t> class tstorage, class tscalar, size_t trank>
struct is_tensor_vector<tensor_t<tstorage, tscalar, trank>>
    : std::conditional<trank == 1U, std::true_type, std::false_type>::type
{
};

template <class T>
inline constexpr bool is_matrix_v = is_eigen_v<T> || is_tensor_matrix<T>::value;

template <class T>
inline constexpr bool is_vector_v = is_eigen_v<T> || is_tensor_vector<T>::value;
} // namespace nano
