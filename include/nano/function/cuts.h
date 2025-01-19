#pragma once

#include <nano/function.h>
#include <nano/function/traits.h>

namespace nano
{
///
/// \brief proxy object to model the left-handside multiplication of a matrix or vector with the variable of a function,
///     useful for easily defining constraints.
///
template <class ttensor, std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true>
struct lhs_multiplied_variable_t
{
    template <class tlambda>
    bool call_scalar(const tlambda& lambda) const
    {
        const auto& a = m_tensor;
        if constexpr (is_eigen_v<ttensor>)
        {
            return a.rows() == size() && a.cols() == 1 && constrain(lambda(a.transpose()));
        }
        else
        {
            static_assert(ttensor::rank() == 1U);

            return a.size() == size() && constrain(lambda(a.transpose()));
        }
    }

    template <class tvector, class tlambda, std::enable_if_t<is_vector_v<tvector>, bool> = true>
    bool call_vector(const tvector& b, const tlambda& lambda) const
    {
        const auto& A = m_tensor;
        if constexpr (is_eigen_v<ttensor>)
        {
            bool ok = A.rows() == b.size() && A.cols() == size();
            for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
            {
                ok = constrain(lambda(A.row(i), b(i)));
            }
            return ok;
        }
        else
        {
            static_assert(ttensor::rank() == 2U);

            bool ok = A.rows() == b.size() && A.cols() == size();
            for (tensor_size_t i = 0; i < A.rows() && ok; ++i)
            {
                ok = constrain(lambda(A.row(i), b(i)));
            }
            return ok;
        }
    }

    tensor_size_t size() const { return m_variable.m_function.size(); }

    bool constrain(constraint_t&& constrain) const { return m_variable.m_function.constrain(std::move(constrain)); }

    // attributes
    const ttensor&      m_tensor;   ///<
    function_variable_t m_variable; ///<
};

template <class ttensor, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true>
auto operator*(const ttensor& tensor, const function_variable_t& variable)
{
    return lhs_multiplied_variable_t<ttensor>{tensor, variable};
}

///
/// \brief register a linear equality constraint: A * x = b.
///
template <class ttensor, class tvector, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true,
          std::enable_if_t<is_vector_v<tvector>, bool>                         = true>
bool operator==(const lhs_multiplied_variable_t<ttensor>& lhs_multiplied_variable, const tvector& vb)
{
    const auto op = [&](const auto& a, const scalar_t b) { return constraint::linear_equality_t{a, -b}; };
    return lhs_multiplied_variable.call_vector(vb, op);
}

///
/// \brief register a linear equality constraint: A * x <= b.
///
template <class ttensor, class tvector, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true,
          std::enable_if_t<is_vector_v<tvector>, bool>                         = true>
bool operator<=(const lhs_multiplied_variable_t<ttensor>& lhs_multiplied_variable, const tvector& vb)
{
    const auto op = [&](const auto& a, const scalar_t b) { return constraint::linear_inequality_t{a, -b}; };
    return lhs_multiplied_variable.call_vector(vb, op);
}

///
/// \brief register a linear equality constraint: A * x >= b.
///
template <class ttensor, class tvector, ///<
          std::enable_if_t<is_matrix_v<ttensor> || is_vector_v<ttensor>, bool> = true,
          std::enable_if_t<is_vector_v<tvector>, bool>                         = true>
bool operator>=(const lhs_multiplied_variable_t<ttensor>& lhs_multiplied_variable, const tvector& vb)
{
    const auto op = [&](const auto& a, const scalar_t b) { return constraint::linear_inequality_t{-a, b}; };
    return lhs_multiplied_variable.call_vector(vb, op);
}

///
/// \brief register a linear equality constraint: a.dot(x) = b.
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator==(const lhs_multiplied_variable_t<tvector>& lhs_multiplied_variable, const scalar_t b)
{
    const auto op = [&](const auto& a) { return constraint::linear_equality_t{a, -b}; };
    return lhs_multiplied_variable.call_scalar(op);
}

///
/// \brief register a linear equality constraint: a.dot(x) <= b.
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator<=(const lhs_multiplied_variable_t<tvector>& lhs_multiplied_variable, const scalar_t b)
{
    const auto op = [&](const auto& a) { return constraint::linear_inequality_t{a, -b}; };
    return lhs_multiplied_variable.call_scalar(op);
}

///
/// \brief register a linear equality constraint: a.dot(x) >= b.
///
template <class tvector, ///<
          std::enable_if_t<is_vector_v<tvector>, bool> = true>
bool operator>=(const lhs_multiplied_variable_t<tvector>& lhs_multiplied_variable, const scalar_t b)
{
    const auto op = [&](const auto& a) { return constraint::linear_inequality_t{-a, b}; };
    return lhs_multiplied_variable.call_scalar(op);
}
} // namespace nano
