#include <nano/core/tokenizer.h>
#include <nano/critical.h>
#include <nano/datasource/tabular.h>

using namespace nano;

tabular_datasource_t::tabular_datasource_t(string_t id, csvs_t csvs, features_t features)
    : tabular_datasource_t(std::move(id), std::move(csvs), std::move(features), string_t::npos)
{
}

tabular_datasource_t::tabular_datasource_t(string_t id, csvs_t csvs, features_t features, size_t target)
    : datasource_t(std::move(id))
    , m_csvs(std::move(csvs))
    , m_features(std::move(features))
    , m_target(target)
{
}

rdatasource_t tabular_datasource_t::clone() const
{
    return std::make_unique<tabular_datasource_t>(*this);
}

void tabular_datasource_t::do_load()
{
    const auto basedir = parameter("datasource::basedir").value<string_t>();

    // check features
    critical(!m_features.empty(), "datasource[", type_id(), "]: need to set at least one feature!");

    for (const auto& feature : m_features)
    {
        critical(feature.type() != feature_type::mclass, "datasource[", type_id(),
                 "]: multi-label features are not supported (", feature, ")!");

        critical(::nano::size(feature.dims()) == 1, "datasource[", type_id(),
                 "]: structured features are not supported (", feature, ")!");
    }

    critical(m_target == string_t::npos || m_target < m_features.size(), "datasource[", type_id(),
             "]: the target feature index (", m_target, ") is not valid, expecting in the [0, ", m_features.size(),
             ") range!");

    // allocate storage
    tensor_size_t samples = 0;
    for (const auto& csv : m_csvs)
    {
        csv.parse(basedir,
                  [&](const string_t&, const tensor_size_t)
                  {
                      ++samples;
                      return true;
                  });
    }

    critical(samples > 0, "datasource[", type_id(), "]: no data to read, check paths!");

    resize(samples, m_features, m_target);

    // load data
    tensor_size_t sample = 0;
    for (const auto& csv : m_csvs)
    {
        log_info("[", type_id(), "]: reading ", csv.m_path, "...");

        const auto old_sample = sample;
        csv.parse(basedir,
                  [&](const string_t& line, tensor_size_t line_index)
                  {
                      this->parse(csv, line, line_index, sample++);
                      return true;
                  });

        const auto samples_read = sample - old_sample;
        critical(csv.m_expected == 0 || samples_read == csv.m_expected, "datasource[", type_id(), "]: read ",
                 samples_read, ", expecting ", csv.m_expected, " samples!");

        datasource_t::testing(make_range(old_sample + csv.m_testing.begin(), old_sample + csv.m_testing.end()));

        log_info("[", type_id(), "]: read ", sample, " samples!");
    }

    critical(sample == samples, "datasource[", type_id(), "]: read ", sample, " samples, expecting ", samples, "!");
}

void tabular_datasource_t::parse(const csv_t& csv, const string_t& line, tensor_size_t line_index, tensor_size_t sample)
{
    critical(sample < samples(), "datasource[", type_id(), "]: too many samples, expecting ", samples(), "!");

    for (auto tokenizer = tokenizer_t{line, csv.m_delim.c_str()}; tokenizer; ++tokenizer)
    {
        critical(tokenizer.count() <= m_features.size(), "datasource[", type_id(), "]: invalid line [", line, "]@",
                 csv.m_path, ":", line_index, ", got ", tokenizer.count(), " tokens, expecting ", m_features.size(),
                 "!");

        const auto  f       = tokenizer.count() - 1;
        const auto  token   = tokenizer.get();
        const auto& feature = m_features[f];

        if (token != csv.m_placeholder)
        {
            try
            {
                set(sample, static_cast<tensor_size_t>(f), token);
            }
            catch (const std::exception&)
            {
                raise("datasource[", type_id(), "]: invalid line [", line, "]@", csv.m_path, ":", line_index,
                      ", invalid token [", token, "] for feature (", feature, ")!");
            }
        }
    }
}
