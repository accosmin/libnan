#include <cassert>
#include <fstream>
#include <nano/core/cmdline.h>
#include <nano/core/tokenizer.h>
#include <nano/critical.h>
#include <nano/version.h>

using namespace nano;

namespace
{
auto valid_option_name(const std::string_view name)
{
    if (starts_with(name, "--"))
    {
        // --name
        return name.size() > 2U && name[2U] != '-';
    }
    else if (starts_with(name, "-"))
    {
        // -n or -name
        return name.size() > 1U;
    }
    else
    {
        return false;
    }
}

auto describe(const cmdoption_t& option)
{
    string_t str = option.m_keywords;
    if (option.m_default_value)
    {
        str += "(";
        str += option.m_default_value.value();
        str += ")";
    }

    return str;
}
} // namespace

cmdresult_t::cmdresult_t() = default;

cmdresult_t::cmdresult_t(const cmdvalues_t& values)
{
    for (const auto& [name, value] : values)
    {
        if (value.has_value())
        {
            m_values[name] = value;
        }
    }
}

bool cmdresult_t::has(const std::string_view name) const
{
    // FIXME: no need to create a string when moving to C++20!
    return m_values.find(string_t{name}) != m_values.end();
}

bool cmdresult_t::has_value(const std::string_view name) const
{
    // FIXME: no need to create a string when moving to C++20!
    const auto it = m_values.find(string_t{name});
    return it != m_values.end() && it->second.has_value();
}

string_t cmdresult_t::get(const std::string_view name) const
{
    // FIXME: no need to create a string when moving to C++20!
    const auto it = m_values.find(string_t{name});
    critical(it != m_values.end(), "cmdline: unrecognized option [", name, "]");
    critical(it->second.has_value(), "cmdline: no value provided for option [", name, "]");
    return it->second.value();
}

cmdconfig_t::cmdconfig_t(const cmdresult_t& options, logger_t logger)
    : m_options(options)
    , m_logger(std::move(logger))
{
    for (const auto& [param_name, param_value] : m_options.m_values)
    {
        if (param_value.is_extra())
        {
            m_params_usage[param_name] = 0;
        }
    }
}

void cmdconfig_t::setup(configurable_t& configurable)
{
    for (const auto& [param_name, param_value] : m_options.m_values)
    {
        if (param_value.has_value() && param_value.is_extra())
        {
            const auto* const data = param_name.data();
            const auto        size = param_name.size();
            const auto        name = starts_with(param_name, "--") ? std::string_view{data + 2U, size - 2U}
                                   : starts_with(param_name, "-")  ? std::string_view{data + 1U, size - 1U}
                                                                   : std::string_view{};

            if (configurable.parameter_if(name) != nullptr)
            {
                configurable.parameter(name) = param_value.value();
                m_params_usage[param_name]++;
            }
        }
    }
}

cmdconfig_t::~cmdconfig_t()
{
    for (const auto& [param_name, count] : m_params_usage)
    {
        if (count == 0)
        {
            m_logger.log(log_type::warn, "parameter '", param_name, "' was not used.\n");
        }
    }
}

cmdline_t::cmdline_t(string_t title)
    : m_title(std::move(title))
{
    add("-h,--help", "print usage");
    add("-v,--version", "print library's version");
    add("-g,--git-hash", "print library's git commit hash");
}

void cmdline_t::add(string_t keywords, string_t description)
{
    add(cmdoption_t{std::move(keywords), std::move(description), ostring_t{}});
}

void cmdline_t::add(string_t keywords, string_t description, string_t default_value)
{
    add(cmdoption_t{std::move(keywords), std::move(description), std::move(default_value)});
}

void cmdline_t::add(cmdoption_t option)
{
    critical(!option.m_keywords.empty(), "cmdline: option cannot be empty");
    critical(!option.m_description.empty(), "cmdline: description cannot be empty");

    for (auto tokenizer = tokenizer_t{option.m_keywords, ","}; tokenizer; ++tokenizer)
    {
        const auto name = tokenizer.get();
        critical(valid_option_name(name), "cmdline: option '", name,
                 "' must start with either a single or a double dash");

        const auto uniq = m_values.emplace(name, cmdvalue_t{option.m_default_value, m_options.size()}).second;
        critical(uniq, "cmdline: duplicated option '", name, "'");
    }

    m_options.emplace_back(std::move(option));
}

cmdresult_t cmdline_t::process(const int argc, const char* argv[]) const
{
    auto result = cmdresult_t{m_values};

    for (int i = 1; i < argc;)
    {
        const auto name = string_t{argv[i]};
        assert(!name.empty());

        critical(valid_option_name(name), "cmdline: expecting option name [", name,
                 "] to start with either a single or a double dash");

        const auto itval = m_values.find(name);
        const auto nextv = (i + 1 < argc) && !valid_option_name(argv[i + 1]);

        // known option...
        if (itval != m_values.end())
        {
            const auto index = itval->second.m_index;
            assert(index < m_options.size());

            // NB: also add all equivalent keywords from the associated option!
            const auto& option = m_options[index];
            for (auto tokenizer = tokenizer_t{option.m_keywords, ","}; tokenizer; ++tokenizer)
            {
                const auto tname       = string_t{tokenizer.get()};
                result.m_values[tname] = nextv ? cmdvalue_t{argv[i + 1], index} : cmdvalue_t{{}, index};
            }
        }

        // unknown option...
        else
        {
            result.m_values[name] = nextv ? cmdvalue_t{argv[i + 1]} : cmdvalue_t{};
        }

        if (nextv)
        {
            // NB: skip two tokens: --option value
            i += 2;
        }
        else
        {
            // NB: skip one token (potentially boolean option): --option
            i += 1;
        }
    }

    return result;
}

cmdresult_t cmdline_t::process(const string_t& config) const
{
    strings_t tokens;
    for (auto tokenizer = tokenizer_t{config, " \t\n\r"}; tokenizer; ++tokenizer)
    {
        tokens.emplace_back(tokenizer.get());
    }

    std::vector<const char*> ptokens(tokens.size() + 1, nullptr);
    std::transform(tokens.begin(), tokens.end(), ptokens.begin() + 1, [&](const auto& token) { return token.c_str(); });

    return process(static_cast<int>(ptokens.size()), ptokens.data());
}

cmdresult_t cmdline_t::process_config_file(const std::filesystem::path& path) const
{
    auto       stream = std::ifstream{path};
    const auto config = string_t{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};

    return process(config);
}

void cmdline_t::usage(std::ostream& stream, const size_t indent) const
{
    stream << m_title << "\n";

    size_t max_option_size = 0;
    for (const auto& option : m_options)
    {
        max_option_size = std::max(max_option_size, describe(option).size());
    }

    const auto header = string_t(indent, ' ');

    max_option_size += 4;
    for (const auto& option : m_options)
    {
        stream << header << nano::align(describe(option), max_option_size) << option.m_description << "\n";
    }
}

bool cmdline_t::handle(const cmdresult_t& options, std::ostream& stream, const size_t indent) const
{
    if (options.has("--help"))
    {
        usage(stream, indent);
        return true;
    }
    else if (options.has("--version"))
    {
        stream << nano::major_version << "." << nano::minor_version << "." << nano::patch_version << "\n";
        return true;
    }
    else if (options.has("--git-hash"))
    {
        stream << nano::git_commit_hash << "\n";
        return true;
    }
    else
    {
        return false;
    }
}
