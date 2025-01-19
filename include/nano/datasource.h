#pragma once

#include <nano/configurable.h>
#include <nano/datasource/storage.h>
#include <nano/factory.h>
#include <nano/loggable.h>

namespace nano
{
class datasource_t;
using rdatasource_t = std::unique_ptr<datasource_t>;

///
/// \brief machine learning dataset consisting of a collection of iid samples.
///
/// NB: each sample consists of:
///     - a fixed number of (input) feature values and
///     - optionally a target if a supervised ML task.
///
/// NB: the input features and the target feature can be optional.
/// NB: the categorical features can be single-label or multi-label.
/// NB: the continuous features can be structured (multi-dimensional) if feature_t::dims() != (1, 1, 1).
///
class NANO_PUBLIC datasource_t : public typed_t,
                                 public configurable_t,
                                 public loggable_t,
                                 public clonable_t<datasource_t>
{
public:
    ///
    /// \brief default constructor
    ///
    explicit datasource_t(string_t id);

    ///
    /// \brief returns the available implementations.
    ///
    static factory_t<datasource_t>& all();

    ///
    /// \brief load dataset in memory.
    ///
    /// NB: any error is considered critical and an exception will be triggered.
    ///
    void load();

    ///
    /// \brief returns the appropriate mathine learning task (by inspecting the target feature).
    ///
    task_type type() const;

    ///
    /// \brief returns the total number of samples.
    ///
    tensor_size_t samples() const { return m_testing.size(); }

    ///
    /// \brief returns the samples that can be used for training.
    ///
    indices_t train_samples() const;

    ///
    /// \brief returns the samples that should only be used for testing.
    ///
    /// NB: assumes a fixed set of test samples.
    ///
    indices_t test_samples() const;

    ///
    /// \brief set all the samples for training.
    ///
    void no_testing();

    ///
    /// \brief set the given range of samples for testing.
    ///
    /// NB: this accumulates the previous range of samples set for testing.
    ///
    void testing(tensor_range_t sample_range);

    ///
    /// \brief returns the total number of features.
    ///
    tensor_size_t features() const
    {
        const auto total = m_storage_range.size<0>();
        return (m_target < total) ? (total - 1) : total;
    }

    ///
    /// \brief returns the feature at the given index.
    ///
    const feature_t& feature(const tensor_size_t ifeature) const
    {
        assert(ifeature >= 0 && ifeature < features());
        return m_features[static_cast<size_t>(ifeature >= m_target ? ifeature + 1 : ifeature)];
    }

    ///
    /// \brief call and return the result of the given operator on the target feature.
    ///
    /// NB: the signature of the operator is: op(feature_t, tensor_cmap_t<> data, mask_cmap_t).
    ///
    template <class toperator>
    auto visit_target(const toperator& op) const
    {
        assert(has_target());
        return visit(m_target, op);
    }

    ///
    /// \brief call and return the result of the given operator on the given feature index.
    ///
    /// NB: the signature of the operator is: op(feature_t, tensor_cmap_t<> data, mask_cmap_t).
    ///
    template <class toperator>
    auto visit_inputs(const tensor_size_t ifeature, const toperator& op) const
    {
        assert(ifeature >= 0 && ifeature < features());
        return visit(ifeature >= m_target ? ifeature + 1 : ifeature, op);
    }

protected:
    ///
    /// \brief allocate the dataset to store the given number of samples and samples.
    ///
    /// NB: no target feature is given
    ///     and as such the dataset represents an unsupervised ML task.
    ///
    void resize(tensor_size_t samples, const features_t& features);

    ///
    /// \brief allocate the dataset to store the given number of samples and samples.
    ///
    /// NB: the target feature is given as an index in the list of features
    ///     and as such the dataset represents a supervised ML task.
    ///
    void resize(tensor_size_t samples, const features_t& features, size_t target);

    ///
    /// \brief safely write a feature value for the given sample.
    ///
    template <class tvalue>
    void set(const tensor_size_t sample, const tensor_size_t ifeature, const tvalue& value)
    {
        assert(sample >= 0 && sample < samples());
        assert(ifeature >= 0 && ifeature < m_storage_range.size<0>());

        this->visit(ifeature,
                    [sample, &value](const feature_t& feature, const auto& data, const auto& mask)
                    {
                        const auto setter = feature_storage_t{feature};
                        setter.set(data, sample, value);
                        setbit(mask, sample);
                    });
    }

private:
    virtual void do_load() = 0;

    bool has_target() const { return m_target < m_storage_range.size<0>(); }

    mask_map_t mask(const tensor_size_t index) { return m_storage_mask.tensor(index); }

    mask_cmap_t mask(const tensor_size_t index) const { return m_storage_mask.tensor(index); }

    static constexpr auto maxu08 = tensor_size_t(1) << 8;
    static constexpr auto maxu16 = tensor_size_t(1) << 16;
    static constexpr auto maxu32 = tensor_size_t(1) << 32;

    template <class toperator>
    auto visit(const tensor_size_t ifeature, const toperator& op)
    {
        const auto& feature = m_features[static_cast<size_t>(ifeature)];

        const auto samples      = this->samples();
        const auto mask         = this->mask(ifeature);
        const auto [d0, d1, d2] = feature.dims();
        const auto range        = make_range(m_storage_range(ifeature, 0), m_storage_range(ifeature, 1));

        switch (feature.type())
        {
        case feature_type::sclass:
            return (feature.classes() <= maxu08) ? op(feature, m_storage_u08.slice(range).reshape(-1), mask)
                 : (feature.classes() <= maxu16) ? op(feature, m_storage_u16.slice(range).reshape(-1), mask)
                 : (feature.classes() <= maxu32) ? op(feature, m_storage_u32.slice(range).reshape(-1), mask)
                                                 : op(feature, m_storage_u64.slice(range).reshape(-1), mask);
        case feature_type::mclass: return op(feature, m_storage_u08.slice(range).reshape(samples, -1), mask);
        case feature_type::float32: return op(feature, m_storage_f32.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::float64: return op(feature, m_storage_f64.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int8: return op(feature, m_storage_i08.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int16: return op(feature, m_storage_i16.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int32: return op(feature, m_storage_i32.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int64: return op(feature, m_storage_i64.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint8: return op(feature, m_storage_u08.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint16: return op(feature, m_storage_u16.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint32: return op(feature, m_storage_u32.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint64: return op(feature, m_storage_u64.slice(range).reshape(samples, d0, d1, d2), mask);
        default: raise("in-memory dataset: unhandled feature type (", static_cast<int>(feature.type()), ")!");
        }
        return op(feature, m_storage_u08.slice(range).reshape(-1), mask);
    }

    template <class toperator>
    auto visit(const tensor_size_t ifeature, const toperator& op) const
    {
        const auto& feature = m_features[static_cast<size_t>(ifeature)];

        const auto samples      = this->samples();
        const auto mask         = this->mask(ifeature);
        const auto [d0, d1, d2] = feature.dims();
        const auto range        = make_range(m_storage_range(ifeature, 0), m_storage_range(ifeature, 1));

        switch (feature.type())
        {
        case feature_type::sclass:
            return (feature.classes() <= maxu08) ? op(feature, m_storage_u08.slice(range).reshape(-1), mask)
                 : (feature.classes() <= maxu16) ? op(feature, m_storage_u16.slice(range).reshape(-1), mask)
                 : (feature.classes() <= maxu32) ? op(feature, m_storage_u32.slice(range).reshape(-1), mask)
                                                 : op(feature, m_storage_u64.slice(range).reshape(-1), mask);
        case feature_type::mclass: return op(feature, m_storage_u08.slice(range).reshape(samples, -1), mask);
        case feature_type::float32: return op(feature, m_storage_f32.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::float64: return op(feature, m_storage_f64.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int8: return op(feature, m_storage_i08.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int16: return op(feature, m_storage_i16.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int32: return op(feature, m_storage_i32.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::int64: return op(feature, m_storage_i64.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint8: return op(feature, m_storage_u08.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint16: return op(feature, m_storage_u16.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint32: return op(feature, m_storage_u32.slice(range).reshape(samples, d0, d1, d2), mask);
        case feature_type::uint64: return op(feature, m_storage_u64.slice(range).reshape(samples, d0, d1, d2), mask);
        default: raise("in-memory dataset: unhandled feature type (", static_cast<int>(feature.type()), ")!");
        }
        return op(feature, m_storage_u08.slice(range).reshape(-1), mask);
    }

    indices_t filter(tensor_size_t count, tensor_size_t condition) const;

    template <class tscalar>
    using storage_t       = tensor_mem_t<tscalar, 2>;
    using storage_mask_t  = tensor_mem_t<uint8_t, 2>;
    using storage_type_t  = std::vector<feature_type>;
    using storage_range_t = tensor_mem_t<tensor_size_t, 2>;

    // attributes
    indices_t           m_testing;       ///< (#samples,) - mark sample for testing, if != 0
    features_t          m_features;      ///< input and target features
    tensor_size_t       m_target{0};     ///< index of the target feature if it exists, otherwise string_t::npos
    storage_t<float>    m_storage_f32;   ///<
    storage_t<double>   m_storage_f64;   ///<
    storage_t<int8_t>   m_storage_i08;   ///<
    storage_t<int16_t>  m_storage_i16;   ///<
    storage_t<int32_t>  m_storage_i32;   ///<
    storage_t<int64_t>  m_storage_i64;   ///<
    storage_t<uint8_t>  m_storage_u08;   ///<
    storage_t<uint16_t> m_storage_u16;   ///<
    storage_t<uint32_t> m_storage_u32;   ///<
    storage_t<uint64_t> m_storage_u64;   ///<
    storage_mask_t      m_storage_mask;  ///< feature value given if the bit (feature, sample) is 1
    storage_type_t      m_storage_type;  ///<
    storage_range_t     m_storage_range; ///<
};
} // namespace nano
