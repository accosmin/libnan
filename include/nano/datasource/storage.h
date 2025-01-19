#pragma once

#include <nano/critical.h>
#include <nano/datasource/mask.h>
#include <nano/feature.h>

namespace nano
{
///
/// \brief utility to safely access feature values.
///
/// a feature value to write can be of a variety of types:
///     - a scalar,
///     - a label index (if single-label categorical),
///     - a label hit vector (if multi-label categorical),
///     - a 3D tensor (if structured continuous) or
///     - a string.
///
class feature_storage_t
{
public:
    ///
    /// \brief constructor.
    ///
    explicit feature_storage_t(const feature_t& feature)
        : m_feature(feature)
    {
    }

    const feature_t& feature() const { return m_feature; }

    tensor3d_dims_t dims() const { return m_feature.dims(); }

    const string_t& name() const { return m_feature.name(); }

    tensor_size_t classes() const { return m_feature.classes(); }

    ///
    /// \brief set the feature value of a sample for a single-label categorical feature.
    ///
#ifdef _WIN32
    #pragma warning(push)
    #pragma warning(disable : 4702) // NB: unreachable code for the default switch label below!
#endif
    template <class tscalar, class tvalue>
    void set(const tensor_map_t<tscalar, 1>& data, tensor_size_t sample, const tvalue& value) const
    {
        tensor_size_t label = 0; // NOLINT(misc-const-correctness)
        if constexpr (std::is_same_v<tvalue, string_t> || std::is_same_v<tvalue, const char*> ||
                      std::is_same_v<tvalue, std::string_view>)
        {
            label = static_cast<tensor_size_t>(m_feature.set_label(value)); // NOLINT(cert-str34-c)
        }
        else if constexpr (std::is_arithmetic_v<tvalue>)
        {
            label = static_cast<tensor_size_t>(value); // NOLINT(cert-str34-c,bugprone-signed-char-misuse)
        }
        else
        {
            raise("in-memory dataset: cannot set single-label feature <", name(), ">!");
        }

        critical(label >= 0 && label < classes(), "in-memory dataset: cannot set single-label feature <", name(),
                 ">: invalid label ", label, " not in [0, ", classes(), ")!");

        data(sample) = static_cast<tscalar>(label); // NOLINT(cert-str34-c)
    }
#ifdef _WIN32
    #pragma warning(pop)
#endif

    ///
    /// \brief set the feature value of a sample for a multi-label categorical feature.
    ///
    template <class tscalar, class tvalue>
    void set(const tensor_map_t<tscalar, 2>& data, [[maybe_unused]] tensor_size_t sample, const tvalue& value) const
    {
        if constexpr (::nano::is_tensor_v<tvalue>)
        {
            if constexpr (tvalue::rank() == 1)
            {
                critical(value.size() == classes(), "in-memory dataset: cannot set multi-label feature <", name(),
                         ">: invalid number of labels ", value.size(), " vs. ", classes(), "!");

                data.vector(sample) = value.vector().template cast<tscalar>();
            }
            else
            {
                raise("in-memory dataset: cannot set multi-label feature <", name(), ">!");
            }
        }
        else
        {
            raise("in-memory dataset: cannot set multi-label feature <", name(), ">!");
        }
    }

    ///
    /// \brief set the feature value of a sample for a continuous scalar or structured feature.
    ///
    template <class tscalar, class tvalue>
    void set(const tensor_map_t<tscalar, 4>& data, [[maybe_unused]] tensor_size_t sample, const tvalue& value) const
    {
        if constexpr (std::is_same_v<tvalue, string_t> || std::is_same_v<tvalue, const char*> ||
                      std::is_same_v<tvalue, std::string_view>)
        {
            critical(::nano::size(dims()) == 1, "in-memory dataset: cannot set scalar feature <", name(),
                     ">: invalid tensor dimensions ", dims(), "!");

            data(sample) = check_from_string<tscalar>("scalar", value);
        }
        else if constexpr (std::is_arithmetic_v<tvalue>)
        {
            critical(::nano::size(dims()) == 1, "in-memory dataset: cannot set scalar feature <", name(),
                     ">: invalid tensor dimensions ", dims(), "!");

            data(sample) = static_cast<tscalar>(value); // NOLINT(cert-str34-c,bugprone-signed-char-misuse)
        }
        else if constexpr (::nano::is_tensor_v<tvalue>)
        {
            critical(::nano::size(dims()) == value.size(), "in-memory dataset: cannot set scalar feature <", name(),
                     ">: invalid tensor dimensions ", dims(), " vs. ", value.dims(), "!");

            data.vector(sample) = value.vector().template cast<tscalar>();
        }
        else
        {
            raise("in-memory dataset: cannot set scalar feature <", name(), ">!");
        }
    }

private:
    template <class tscalar>
    auto check_from_string(const char* type, const std::string_view& value) const
    {
        tscalar scalar;
        try
        {
            scalar = ::nano::from_string<tscalar>(value);
        }
        catch (const std::exception& e)
        {
            raise("in-memory dataset: cannot set ", type, " feature <", name(), ">: caught exception <", e.what(),
                  ">!");
        }
        return scalar;
    }

    // attributes
    const feature_t& m_feature; ///<
};
} // namespace nano
