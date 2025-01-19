#pragma once

#include <nano/function.h>
#include <nano/function/traits.h>

namespace nano
{
///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] <= upper[i].
///
template <class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator<=(const function_variable_t& variable, const tvector& upper)
{
    bool ok = upper.size() == variable.m_function.size();
    for (tensor_size_t i = 0; i < upper.size() && ok; ++i)
    {
        ok = variable.m_function.constrain(constraint::maximum_t{upper(i), i});
    }
    return ok;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: lower[i] <= x[i].
///
template <class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator<=(const tvector& lower, const function_variable_t& variable)
{
    bool ok = lower.size() == variable.m_function.size();
    for (tensor_size_t i = 0; i < lower.size() && ok; ++i)
    {
        ok = variable.m_function.constrain(constraint::minimum_t{lower(i), i});
    }
    return ok;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] >= lower[i].
///
template <class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator>=(const function_variable_t& variable, const tvector& lower)
{
    return lower <= variable;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: upper[i] >= x[i].
///
template <class tvector, std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator>=(const tvector& upper, const function_variable_t& variable)
{
    return variable <= upper;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] <= upper.
///
inline bool operator<=(const function_variable_t& variable, const scalar_t upper)
{
    return variable <= vector_t::constant(variable.m_function.size(), upper);
}

///
/// \brief register a one-sided inequality contraint for all dimensions: lower <= x[i].
///
inline bool operator<=(const scalar_t lower, const function_variable_t& variable)
{
    return vector_t::constant(variable.m_function.size(), lower) <= variable;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: x[i] >= lower.
///
inline bool operator>=(const function_variable_t& variable, const scalar_t lower)
{
    return lower <= variable;
}

///
/// \brief register a one-sided inequality contraint for all dimensions: upper >= x[i].
///
inline bool operator>=(const scalar_t upper, const function_variable_t& variable)
{
    return variable <= upper;
}

///
/// \brief register a one-sided inequality contraint for the given dimension: x[dimension] <= upper.
///
inline bool operator<=(const function_variable_dimension_t& ivariable, const scalar_t upper)
{
    return ivariable.m_dimension >= 0 && ivariable.m_dimension < ivariable.m_function.size() &&
           ivariable.m_function.constrain(constraint::maximum_t{upper, ivariable.m_dimension});
}

///
/// \brief register a one-sided inequality contraint for the given dimension: lower <= x[dimension].
///
inline bool operator<=(const scalar_t lower, const function_variable_dimension_t& ivariable)
{
    return ivariable.m_dimension >= 0 && ivariable.m_dimension < ivariable.m_function.size() &&
           ivariable.m_function.constrain(constraint::minimum_t{lower, ivariable.m_dimension});
}

///
/// \brief register a one-sided inequality contraint for the given dimension: x[dimension] >= lower.
///
inline bool operator>=(const function_variable_dimension_t& ivariable, const scalar_t lower)
{
    return lower <= ivariable;
}

///
/// \brief register a one-sided inequality contraint for the given dimension: upper >= x[dimension].
///
inline bool operator>=(const scalar_t upper, const function_variable_dimension_t& ivariable)
{
    return ivariable <= upper;
}
} // namespace nano
