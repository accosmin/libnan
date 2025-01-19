#include <deque>
#include <iomanip>
#include <nano/tensor/stream.h>
#include <nano/wlearner/criterion.h>
#include <nano/wlearner/dtree.h>
#include <nano/wlearner/stump.h>
#include <nano/wlearner/util.h>
#include <set>

using namespace nano;

namespace
{
class cache_t
{
public:
    cache_t() = default;

    explicit cache_t(indices_t samples)
        : m_samples(std::move(samples))
    {
    }

    // attributes
    indices_t     m_samples;   ///<
    tensor_size_t m_depth{0};  ///<
    tensor3d_t    m_table;     ///<
    size_t        m_parent{0}; ///<
};

void append(tensor4d_t& tables, const tensor3d_cmap_t& table)
{
    // NB: This conservative resize is not very efficient!
    const auto copy  = tables;
    const auto count = copy.size<0>();
    tables.resize(cat_dims(count + 1, table.dims()));
    if (count > 0)
    {
        tables.slice(0, count) = copy;
    }
    tables.tensor(count) = table;
}

auto unique_features(const dtree_nodes_t& nodes)
{
    // gather the unique set of selected features
    std::set<tensor_size_t> ufeatures;
    for (const auto& node : nodes)
    {
        if (node.m_feature >= 0)
        {
            ufeatures.insert(node.m_feature);
        }
    }

    auto features = indices_t{static_cast<tensor_size_t>(ufeatures.size())};

    auto it = ufeatures.begin();
    for (tensor_size_t i = 0; i < features.size(); ++i)
    {
        features(i) = *(it++);
    }

    return features;
}
} // namespace

std::istream& nano::read(std::istream& stream, dtree_node_t& node)
{
    if (!::nano::read_cast<int32_t>(stream, node.m_feature) || !::nano::read(stream, node.m_threshold) ||
        !::nano::read_cast<uint32_t>(stream, node.m_next) || !::nano::read_cast<int32_t>(stream, node.m_table))
    {
        stream.setstate(std::ios_base::failbit); // LCOV_EXCL_LINE
    }
    return stream;
}

std::ostream& nano::write(std::ostream& stream, const dtree_node_t& node)
{
    if (!::nano::write(stream, static_cast<int32_t>(node.m_feature)) || !::nano::write(stream, node.m_threshold) ||
        !::nano::write(stream, static_cast<uint32_t>(node.m_next)) ||
        !::nano::write(stream, static_cast<int32_t>(node.m_table)))
    {
        stream.setstate(std::ios_base::failbit); // LCOV_EXCL_LINE
    }
    return stream;
}

std::ostream& nano::operator<<(std::ostream& stream, const dtree_node_t& node)
{
    return stream << "node: feature=" << node.m_feature << ",threshold=" << node.m_threshold << ",next=" << node.m_next
                  << ",table=" << node.m_table;
}

std::ostream& nano::operator<<(std::ostream& stream, const dtree_nodes_t& nodes)
{
    stream << "nodes:{\n";
    for (const auto& node : nodes)
    {
        stream << "\t" << node << "\n";
    }
    return stream << "}";
}

dtree_wlearner_t::dtree_wlearner_t()
    : wlearner_t("dtree")
{
    register_parameter(parameter_t::make_integer("wlearner::dtree::max_depth", 1, LE, 3, LE, 10));
    register_parameter(parameter_t::make_integer("wlearner::dtree::min_split", 1, LE, 5, LE, 10));
}

std::istream& dtree_wlearner_t::read(std::istream& stream)
{
    wlearner_t::read(stream);

    critical(::nano::read(stream, m_nodes) && ::nano::read(stream, m_features) && ::nano::read(stream, m_tables),
             "dtree weak learner: failed to read from stream!");

    return stream;
}

std::ostream& dtree_wlearner_t::write(std::ostream& stream) const
{
    wlearner_t::write(stream);

    critical(::nano::write(stream, m_nodes) && ::nano::write(stream, m_features) && ::nano::write(stream, m_tables),
             "dtree weak learner: failed to write to stream!");

    return stream;
}

rwlearner_t dtree_wlearner_t::clone() const
{
    return std::make_unique<dtree_wlearner_t>(*this);
}

void dtree_wlearner_t::scale(const vector_t& scale)
{
    ::nano::wlearner::scale(m_tables, scale);
}

scalar_t dtree_wlearner_t::do_fit(const dataset_t& dataset, const indices_t& samples, const tensor4d_t& gradients)
{
    const auto max_depth = parameter("wlearner::dtree::max_depth").value<tensor_size_t>();
    const auto min_split = parameter("wlearner::dtree::min_split").value<tensor_size_t>();
    const auto criterion = parameter("wlearner::criterion").value<wlearner_criterion>();

    const auto min_samples_size = std::min<tensor_size_t>(10, dataset.samples() * min_split / 100);

    auto score  = scalar_t{0};
    auto nodes  = dtree_nodes_t{};
    auto stump  = stump_wlearner_t{};
    auto tables = tensor4d_t{cat_dims(0, dataset.target_dims())};

    stump.parameter("wlearner::criterion") = criterion;

    std::deque<cache_t> caches;
    caches.emplace_back(samples);
    while (!caches.empty())
    {
        const auto cache = caches.front();

        // split the node using decision stumps...
        log_info('[', type_id(), "]: ", std::fixed, std::setprecision(8), " +++ depth=", cache.m_depth,
                 ",samples=", cache.m_samples.size(),
                 ",score=", score == wlearner_t::no_fit_score() ? scat("N/A") : scat(score), "...\n");
        const auto score_stump = stump.fit(dataset, cache.m_samples, gradients);
        if (score_stump == wlearner_t::no_fit_score())
        {
            score = wlearner_t::no_fit_score();
            break;
        }

        dtree_node_t node;
        node.m_feature   = stump.feature();
        node.m_threshold = stump.threshold();

        const auto& tables_stump = stump.tables();
        const auto  cluster      = stump.split(dataset, cache.m_samples);
        assert(cluster.groups() == tables_stump.size<0>());

        // have the parent node point to the current terminal node (to be added)
        if (cache.m_parent < nodes.size())
        {
            nodes[cache.m_parent].m_next = nodes.size();
        }

        // terminal nodes...
        if (cache.m_samples.size() < min_samples_size || (cache.m_depth + 1) >= max_depth)
        {
            for (tensor_size_t i = 0, size = tables_stump.size<0>(); i < size; ++i)
            {
                node.m_table = tables.size<0>();
                nodes.emplace_back(node);
                append(tables, tables_stump.tensor(i));
            }

            // also, update the total score
            score += score_stump;
        }

        // can still split the samples
        else
        {
            cache_t ncache;
            ncache.m_depth = cache.m_depth + 1;

            for (tensor_size_t i = 0, size = tables_stump.size<0>(); i < size; ++i)
            {
                ncache.m_parent  = nodes.size();
                ncache.m_samples = cluster.indices(i);

                node.m_table = -1;
                nodes.push_back(node);
                caches.push_back(ncache);
            }
        }

        caches.pop_front();
    }

    // OK, compact the selected features
    auto features = unique_features(nodes);

    log_info('[', type_id(), "]: ", std::fixed, std::setprecision(8), " === tree(features=", features.size(),
             ",nodes=", nodes.size(), ",leafs=", tables.size<0>(), ")",
             ",score=", score == wlearner_t::no_fit_score() ? scat("N/A") : scat(score), ".\n");

    if (score != wlearner_t::no_fit_score())
    {
        m_nodes    = std::move(nodes);
        m_tables   = std::move(tables);
        m_features = std::move(features);
    }

    return score;
}

void dtree_wlearner_t::do_predict(const dataset_t& dataset, indices_cmap_t samples, tensor4d_map_t outputs) const
{
    const auto cluster = split(dataset, samples);
    for (tensor_size_t i = 0, size = samples.size(); i < size; ++i)
    {
        const auto sample = samples(i);
        const auto group  = cluster.group(sample);
        if (group >= 0)
        {
            assert(group < m_tables.size<0>());
            outputs.vector(i) += m_tables.vector(group);
        }
    }
}

cluster_t dtree_wlearner_t::do_split(const dataset_t& dataset, const indices_t& samples) const
{
    cluster_t cluster(dataset.samples(), m_tables.size());

    std::deque<std::pair<size_t, indices_t>> splits;
    splits.emplace_back(0U, samples);

    while (!splits.empty())
    {
        const auto split = splits.front();
        assert(split.first < m_nodes.size());

        const auto& node         = m_nodes[split.first];
        const auto& node_samples = split.second;
        const auto  node_cluster = stump_wlearner_t::split(dataset, node_samples, node.m_feature, node.m_threshold);

        // terminal node
        if (node.m_next == 0U)
        {
            for (tensor_size_t sample = 0; sample < node_cluster.samples(); ++sample)
            {
                const auto group = node_cluster.group(sample);
                if (group >= 0)
                {
                    cluster.assign(sample, node.m_table + group);
                }
            }
        }

        // split node
        else
        {
            for (tensor_size_t group = 0; group < node_cluster.groups(); ++group)
            {
                splits.emplace_back(m_nodes[split.first + static_cast<size_t>(group)].m_next,
                                    node_cluster.indices(group));
            }
        }

        splits.pop_front();
    }

    return cluster;
}

indices_t dtree_wlearner_t::features() const
{
    return m_features;
}
