#include <nano/core/stream.h>
#include <nano/critical.h>
#include <nano/feature.h>

using namespace nano;

feature_t::feature_t() = default;

feature_t::feature_t(string_t name)
    : m_name(std::move(name))
{
}

feature_t& feature_t::scalar(feature_type type, tensor3d_dims_t dims)
{
    assert(type != feature_type::sclass && type != feature_type::mclass);

    m_dims = dims;
    m_type = type;
    m_labels.clear();
    return *this;
}

feature_t& feature_t::sclass(strings_t labels)
{
    m_type   = feature_type::sclass;
    m_labels = std::move(labels);
    return *this;
}

feature_t& feature_t::mclass(strings_t labels)
{
    m_type   = feature_type::mclass;
    m_labels = std::move(labels);
    return *this;
}

feature_t& feature_t::sclass(size_t count)
{
    m_type   = feature_type::sclass;
    m_labels = strings_t(count);
    return *this;
}

feature_t& feature_t::mclass(size_t count)
{
    m_type   = feature_type::mclass;
    m_labels = strings_t(count);
    return *this;
}

size_t feature_t::set_label(const char* label) const
{
    return set_label(std::string_view(label));
}

size_t feature_t::set_label(const string_t& label) const
{
    return set_label(std::string_view{label});
}

size_t feature_t::set_label(const std::string_view& label) const
{
    if (label.empty())
    {
        return string_t::npos;
    }

    const auto it = std::find(m_labels.begin(), m_labels.end(), label);
    if (it == m_labels.end())
    {
        // new label, replace the first empty label with it
        for (size_t i = 0; i < m_labels.size(); ++i)
        {
            if (m_labels[i].empty())
            {
                m_labels[i] = label;
                return i;
            }
        }

        // new label, but no new place for it
        return string_t::npos;
    }
    else
    {
        // known label, ignore
        return static_cast<size_t>(std::distance(m_labels.begin(), it));
    }
}

bool feature_t::valid() const
{
    return !m_name.empty();
}

task_type feature_t::task() const
{
    if (!valid())
    {
        return task_type::unsupervised;
    }
    else
    {
        switch (m_type)
        {
        case feature_type::sclass: return task_type::sclassification;
        case feature_type::mclass: return task_type::mclassification;
        default: return task_type::regression;
        }
    }
}

std::istream& feature_t::read(std::istream& stream)
{
    string_t type;
    critical(::nano::read(stream, type) && ::nano::read(stream, m_dims) && ::nano::read(stream, m_name) &&
                 ::nano::read(stream, m_labels),
             "feature (", m_name, "): failed to read from stream!");

    m_type = from_string<feature_type>(type);

    return stream;
}

std::ostream& feature_t::write(std::ostream& stream) const
{
    critical(::nano::write(stream, scat(m_type)) && ::nano::write(stream, m_dims) && ::nano::write(stream, m_name) &&
                 ::nano::write(stream, m_labels),
             "feature (", m_name, "): failed to write to stream!");

    return stream;
}

bool nano::operator==(const feature_t& f1, const feature_t& f2)
{
    return f1.type() == f2.type() && f1.name() == f2.name() && f1.dims() == f2.dims() && f1.labels() == f2.labels();
}

bool nano::operator!=(const feature_t& f1, const feature_t& f2)
{
    return f1.type() != f2.type() || f1.name() != f2.name() || f1.dims() != f2.dims() || f1.labels() != f2.labels();
}

std::ostream& nano::operator<<(std::ostream& stream, const feature_t& feature)
{
    stream << "name=" << feature.name() << ",type=" << feature.type() << ",dims=" << feature.dims();
    if (!feature.labels().empty())
    {
        stream << ",labels=[";
        for (const auto& label : feature.labels())
        {
            stream << label;
            if (&label != &(*(feature.labels().rbegin())))
            {
                stream << ",";
            }
        }
        stream << "]";
    }
    return stream;
}
