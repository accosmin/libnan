#include <iomanip>
#include <nano/core/reduce.h>
#include <nano/tensor/stream.h>
#include <nano/wlearner/accumulator.h>
#include <nano/wlearner/criterion.h>
#include <nano/wlearner/table.h>
#include <nano/wlearner/util.h>

using namespace nano;
using namespace nano::wlearner;

namespace
{
template <class toperator>
void process(const dataset_t& dataset, const indices_cmap_t& samples, const tensor_size_t feature,
             const hashes_t& hashes, const indices_t& hash2tables, const toperator& op)
{
    switch (dataset.feature(feature).type())
    {
    case feature_type::sclass:
        loop_sclass(dataset, samples, feature,
                    [&](const tensor_size_t i, const int32_t value)
                    {
                        const auto index = ::nano::find(hashes, value);
                        if (index >= 0)
                        {
                            assert(index < hash2tables.size());
                            op(i, hash2tables(index));
                        }
                    });
        break;

    default:
        loop_mclass(dataset, samples, feature,
                    [&](const tensor_size_t i, const auto& values)
                    {
                        const auto index = ::nano::find(hashes, values);
                        if (index >= 0)
                        {
                            assert(index < hash2tables.size());
                            op(i, hash2tables(index));
                        }
                    });
        break;
    }
}
} // namespace

class table_wlearner_t::cache_t : public accumulator_t
{
public:
    explicit cache_t(const tensor3d_dims_t& tdims = {0, 0, 0})
        : accumulator_t(tdims)
    {
    }

    auto score(const tensor_size_t bin) const { return (r2(bin) - r1(bin).square() / x0(bin)).sum(); }

    void score_dense(const tensor_size_t feature, const hashes_t& hashes, const wlearner_criterion criterion)
    {
        const auto bins = this->bins();

        auto rss = m_missing_rss;
        for (tensor_size_t bin = 0; bin < bins; ++bin)
        {
            rss += this->score(bin);
        }
        const auto k = bins * ::nano::size(tdims());
        const auto n = m_samples;

        const auto score = make_score(criterion, rss, k, n);
        if (std::isfinite(score) && score < m_score)
        {
            m_score   = score;
            m_hashes  = hashes;
            m_feature = feature;

            m_hash2tables.resize(bins);
            m_tables.resize(cat_dims(bins, tdims()));

            for (tensor_size_t bin = 0; bin < bins; ++bin)
            {
                m_hash2tables(bin)  = bin;
                m_tables.array(bin) = r1(bin) / x0(bin);
            }
        }
    }

    void score_kbest(const tensor_size_t feature, const hashes_t& hashes, const wlearner_criterion criterion,
                     tensor_size_t max_kbest = -1)
    {
        const auto bins    = this->bins();
        const auto mapping = this->sort();

        auto rss = m_missing_rss;
        for (tensor_size_t bin = 0; bin < bins; ++bin)
        {
            rss += r2(bin).sum();
        }

        max_kbest = max_kbest < 1 ? bins : max_kbest;
        for (tensor_size_t kbest = 1; kbest <= max_kbest; ++kbest)
        {
            rss += mapping[static_cast<size_t>(kbest - 1)].first;

            const auto k = kbest * ::nano::size(tdims());
            const auto n = m_samples;

            const auto score = make_score(criterion, rss, k, n);
            if (std::isfinite(score) && score < m_score)
            {
                m_score       = score;
                m_feature     = feature;
                m_hash2tables = arange(0, kbest);

                m_hashes.resize(kbest);
                m_tables.resize(cat_dims(kbest, tdims()));

                for (tensor_size_t fv = 0; fv < kbest; ++fv)
                {
                    const auto bin     = mapping[static_cast<size_t>(fv)].second;
                    m_hashes(fv)       = hashes(bin);
                    m_tables.array(fv) = r1(bin) / x0(bin);
                }
            }
        }
    }

    void score_ksplit(const tensor_size_t feature, const hashes_t& hashes, const wlearner_criterion criterion)
    {
        const auto bins = this->bins();

        const auto [cluster_x0, cluster_r1, cluster_r2, cluster_rx, cluster_id] = this->cluster();

        for (tensor_size_t ic = 0; ic < bins; ++ic)
        {
            const auto ksplit = bins - ic;

            const auto x0 = cluster_x0.tensor(ic);
            const auto r1 = cluster_r1.tensor(ic);
            const auto r2 = cluster_r2.tensor(ic);
            const auto rx = cluster_rx.tensor(ic);
            const auto id = cluster_id.tensor(ic);

            auto rss = m_missing_rss;
            for (tensor_size_t fv = 0; fv < ksplit; ++fv)
            {
                rss += (r2.array(fv) - r1.array(fv).square() / x0(fv)).sum();
            }

            const auto k = ksplit * ::nano::size(tdims());
            const auto n = m_samples;

            const auto score = make_score(criterion, rss, k, n);
            if (std::isfinite(score) && score < m_score)
            {
                m_score       = score;
                m_hashes      = hashes;
                m_feature     = feature;
                m_hash2tables = id;
                m_tables      = rx.slice(0, ksplit);
            }
        }
    }

    template <class tfvalues, class tvalidator>
    auto update(const indices_t& samples, const tensor4d_t& gradients, const tfvalues& fvalues,
                const tvalidator& validator)
    {
        auto       hashes  = make_hashes(fvalues);
        const auto classes = hashes.size();

        clear(classes);
        m_samples     = 0;
        m_missing_rss = 0.0;
        for (tensor_size_t i = 0, size = fvalues.template size<0>(); i < size; ++i)
        {
            if (const auto& [valid, value] = validator(i); valid)
            {
                const auto bin = find(hashes, value);
                assert(bin >= 0 && bin < classes);
                accumulator_t::update(gradients.array(samples(i)), bin);
            }
            else
            {
                m_missing_rss += gradients.array(samples(i)).square().sum();
            }
            ++m_samples;
        }

        return hashes;
    } // LCOV_EXCL_LINE

    auto update(const indices_t& samples, const tensor4d_t& gradients, const sclass_cmap_t& fvalues)
    {
        return update(samples, gradients, fvalues,
                      [&](const tensor_size_t i)
                      {
                          const auto value = fvalues(i);
                          return std::make_pair(value >= 0, value);
                      });
    }

    auto update(const indices_t& samples, const tensor4d_t& gradients, const mclass_cmap_t& fvalues)
    {
        return update(samples, gradients, fvalues,
                      [&](const tensor_size_t i)
                      {
                          const auto value = fvalues.array(i);
                          return std::make_pair(value(0) >= 0, value);
                      });
    }

    // attributes
    tensor_size_t m_samples{0};                        ///<
    tensor4d_t    m_tables;                            ///<
    tensor_size_t m_feature{-1};                       ///<
    scalar_t      m_missing_rss{0.0};                  ///<
    scalar_t      m_score{wlearner_t::no_fit_score()}; ///<
    hashes_t      m_hashes;                            ///<
    indices_t     m_hash2tables;                       ///<
};

table_wlearner_t::table_wlearner_t(string_t id)
    : single_feature_wlearner_t(std::move(id))
{
}

std::istream& table_wlearner_t::read(std::istream& stream)
{
    single_feature_wlearner_t::read(stream);

    critical(::nano::read(stream, m_hashes) && ::nano::read(stream, m_hash2tables),
             "table weak learner: failed to read from stream!");

    return stream;
}

std::ostream& table_wlearner_t::write(std::ostream& stream) const
{
    single_feature_wlearner_t::write(stream);

    critical(::nano::write(stream, m_hashes) && ::nano::write(stream, m_hash2tables),
             "table weak learner: failed to write to stream!");

    return stream;
}

void table_wlearner_t::do_predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const
{
    process(dataset, samples, feature(), m_hashes, m_hash2tables,
            [&](const tensor_size_t i, const tensor_size_t table) { outputs.vector(i) += vector(table); });
}

cluster_t table_wlearner_t::do_split(const dataset_t& dataset, const indices_t& samples) const
{
    const auto& tables  = this->tables();
    const auto  classes = tables.size<0>();

    cluster_t cluster(dataset.samples(), classes);

    process(dataset, samples, feature(), m_hashes, m_hash2tables,
            [&](const tensor_size_t i, const tensor_size_t table) { cluster.assign(samples(i), table); });

    return cluster;
} // LCOV_EXCL_LINE

scalar_t table_wlearner_t::set([[maybe_unused]] const dataset_t& dataset, [[maybe_unused]] const indices_t& samples,
                               const cache_t& cache)
{
    const auto feature = cache.m_feature >= 0 ? dataset.feature(cache.m_feature) : feature_t{};

    log_info('[', type_id(), "]: ", std::fixed, std::setprecision(8), " === table(feature=", cache.m_feature, "|",
             feature.valid() ? feature.name() : string_t("N/A"),
             ",classes=", feature.valid() ? scat(feature.classes()) : string_t("N/A"),
             ",tables=", cache.m_tables.size<0>(), ",hashes=", cache.m_hashes.size(), "),samples=", samples.size(),
             ",score=", cache.m_score == wlearner_t::no_fit_score() ? scat("N/A") : scat(cache.m_score), ".\n");

    if (cache.m_score != wlearner_t::no_fit_score())
    {
        single_feature_wlearner_t::set(cache.m_feature, cache.m_tables);

        m_hashes      = cache.m_hashes;
        m_hash2tables = cache.m_hash2tables;
    }

    return cache.m_score;
}

dense_table_wlearner_t::dense_table_wlearner_t()
    : table_wlearner_t("dense-table")
{
}

rwlearner_t dense_table_wlearner_t::clone() const
{
    return std::make_unique<dense_table_wlearner_t>(*this);
}

scalar_t dense_table_wlearner_t::do_fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    const auto criterion = parameter("wlearner::criterion").value<wlearner_criterion>();
    const auto iterator  = select_iterator_t{dataset};

    std::vector<cache_t> caches(iterator.concurrency(), cache_t{dataset.target_dims()});
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, sclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_dense(feature, hashes, criterion);
                  });
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, mclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_dense(feature, hashes, criterion);
                  });

    // OK, return and store the optimum feature across threads
    return table_wlearner_t::set(dataset, samples, min_reduce(caches));
}

kbest_table_wlearner_t::kbest_table_wlearner_t()
    : table_wlearner_t("kbest-table")
{
}

rwlearner_t kbest_table_wlearner_t::clone() const
{
    return std::make_unique<kbest_table_wlearner_t>(*this);
}

scalar_t kbest_table_wlearner_t::do_fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    const auto criterion = parameter("wlearner::criterion").value<wlearner_criterion>();
    const auto iterator  = select_iterator_t{dataset};

    std::vector<cache_t> caches(iterator.concurrency(), cache_t{dataset.target_dims()});
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, sclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_kbest(feature, hashes, criterion);
                  });
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, mclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_kbest(feature, hashes, criterion);
                  });

    // OK, return and store the optimum feature across threads
    return table_wlearner_t::set(dataset, samples, min_reduce(caches));
}

ksplit_table_wlearner_t::ksplit_table_wlearner_t()
    : table_wlearner_t("ksplit-table")
{
}

rwlearner_t ksplit_table_wlearner_t::clone() const
{
    return std::make_unique<ksplit_table_wlearner_t>(*this);
}

scalar_t ksplit_table_wlearner_t::do_fit(const dataset_t& dataset, const indices_t& samples,
                                         const tensor4d_t& gradients)
{
    const auto criterion = parameter("wlearner::criterion").value<wlearner_criterion>();
    const auto iterator  = select_iterator_t{dataset};

    std::vector<cache_t> caches(iterator.concurrency(), cache_t{dataset.target_dims()});
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, sclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_ksplit(feature, hashes, criterion);
                  });
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, mclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_ksplit(feature, hashes, criterion);
                  });

    // OK, return and store the optimum feature across threads
    return table_wlearner_t::set(dataset, samples, min_reduce(caches));
}

dstep_table_wlearner_t::dstep_table_wlearner_t()
    : table_wlearner_t("dstep-table")
{
}

rwlearner_t dstep_table_wlearner_t::clone() const
{
    return std::make_unique<dstep_table_wlearner_t>(*this);
}

scalar_t dstep_table_wlearner_t::do_fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    const auto criterion = parameter("wlearner::criterion").value<wlearner_criterion>();
    const auto iterator  = select_iterator_t{dataset};

    std::vector<cache_t> caches(iterator.concurrency(), cache_t{dataset.target_dims()});
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, sclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_kbest(feature, hashes, criterion, 1);
                  });
    iterator.loop(samples,
                  [&](const tensor_size_t feature, const size_t tnum, mclass_cmap_t fvalues)
                  {
                      auto&      cache  = caches[tnum];
                      const auto hashes = cache.update(samples, gradients, fvalues);
                      cache.score_kbest(feature, hashes, criterion, 1);
                  });

    // OK, return and store the optimum feature across threads
    return table_wlearner_t::set(dataset, samples, min_reduce(caches));
}

bool table_wlearner_t::try_merge(const rwlearner_t& other)
{
    if (const auto* const pother = dynamic_cast<const table_wlearner_t*>(other.get()); pother != nullptr)
    {
        if (hashes() == pother->hashes() && hash2tables() == pother->hash2tables())
        {
            return single_feature_wlearner_t::do_try_merge(pother->feature(), pother->tables());
        }
    }
    return false;
}
