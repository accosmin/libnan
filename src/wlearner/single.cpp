#include <nano/critical.h>
#include <nano/tensor/stream.h>
#include <nano/wlearner/single.h>
#include <nano/wlearner/util.h>

using namespace nano;

single_feature_wlearner_t::single_feature_wlearner_t(string_t id)
    : wlearner_t(std::move(id))
{
}

void single_feature_wlearner_t::set(const tensor_size_t feature, const tensor4d_t& tables)
{
    m_tables  = tables;
    m_feature = feature;
}

std::istream& single_feature_wlearner_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(::nano::read_cast<int64_t>(stream, m_feature) && ::nano::read(stream, m_tables),
             "single feature weak learner: failed to read from stream!");

    return stream;
}

std::ostream& single_feature_wlearner_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(::nano::write(stream, static_cast<int64_t>(m_feature)) && ::nano::write(stream, m_tables),
             "single feature learner: failed to write to stream!");

    return stream;
}

void single_feature_wlearner_t::scale(const vector_t& scale)
{
    ::nano::wlearner::scale(m_tables, scale);
}

indices_t single_feature_wlearner_t::features() const
{
    return make_tensor<tensor_size_t>(make_dims(1), m_feature);
}

bool single_feature_wlearner_t::do_try_merge(const tensor_size_t feature, const tensor4d_t& tables)
{
    if (m_feature == feature && m_tables.dims() == tables.dims())
    {
        m_tables.vector() += tables.vector();
        return true;
    }
    return false;
}
