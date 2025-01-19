#include <nano/configurable.h>
#include <nano/core/stream.h>
#include <nano/critical.h>

using namespace nano;

namespace
{
parameter_t* find_param(parameters_t& parameters, const std::string_view name, const bool mandatory)
{
    const auto it = std::find_if(parameters.begin(), parameters.end(),
                                 [&](const parameter_t& param) { return param.name() == name; });

    critical(!mandatory || it != parameters.end(), "configurable: cannot find mandatory parameter (", name, ")!");

    return (it == parameters.end()) ? nullptr : (&*it);
}

const parameter_t* find_param(const parameters_t& parameters, const std::string_view name, const bool mandatory)
{
    const auto it = std::find_if(parameters.begin(), parameters.end(),
                                 [&](const parameter_t& param) { return param.name() == name; });

    critical(!mandatory || it != parameters.end(), "configurable: cannot find mandatory parameter (", name, ")!");

    return (it == parameters.end()) ? nullptr : (&*it);
}
} // namespace

void configurable_t::register_parameter(parameter_t parameter)
{
    critical(!parameter_if(parameter.name()), "configurable: cannot register duplicated parameter (", parameter.name(),
             ")!");

    m_parameters.emplace_back(std::move(parameter));
}

parameter_t& configurable_t::parameter(const std::string_view name)
{
    return *find_param(m_parameters, name, true);
}

const parameter_t& configurable_t::parameter(const std::string_view name) const
{
    return *find_param(m_parameters, name, true);
}

parameter_t* configurable_t::parameter_if(const std::string_view name)
{
    return find_param(m_parameters, name, false);
}

const parameter_t* configurable_t::parameter_if(const std::string_view name) const
{
    return find_param(m_parameters, name, false);
}

std::istream& configurable_t::read(std::istream& stream)
{
    critical(::nano::read(stream, m_major_version) && ::nano::read(stream, m_minor_version) &&
                 ::nano::read(stream, m_patch_version),
             "configurable: failed to read version from stream!");

    critical(m_major_version < nano::major_version ||
                 (m_major_version == nano::major_version && m_minor_version < nano::minor_version) ||
                 (m_major_version == nano::major_version && m_minor_version == nano::minor_version &&
                  m_patch_version <= nano::patch_version),
             "configurable: version mismatch!");

    critical(::nano::read(stream, m_parameters), "configurable: failed to read parameters from stream!");

    return stream;
}

std::ostream& configurable_t::write(std::ostream& stream) const
{
    critical(::nano::write(stream, nano::major_version) && ::nano::write(stream, nano::minor_version) &&
                 ::nano::write(stream, nano::patch_version),
             "configurable: failed to write version to stream");

    critical(::nano::write(stream, m_parameters), "configurable: failed to write parameters to stream!");

    return stream;
}
