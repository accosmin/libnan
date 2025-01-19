#include <mutex>
#include <nano/core/chrono.h>
#include <nano/datasource/imclass_cifar.h>
#include <nano/datasource/imclass_mnist.h>
#include <nano/datasource/tabular.h>

using namespace nano;

datasource_t::datasource_t(string_t id)
    : typed_t(std::move(id))
{
    register_parameter(parameter_t::make_string("datasource::basedir", ""));
}

indices_t datasource_t::train_samples() const
{
    return filter(samples() - m_testing.vector().sum(), 0);
}

indices_t datasource_t::test_samples() const
{
    return filter(m_testing.vector().sum(), 1);
}

void datasource_t::no_testing()
{
    m_testing.zero();
}

void datasource_t::testing(tensor_range_t sample_range)
{
    assert(sample_range.begin() >= 0 && sample_range.end() <= m_testing.size());
    m_testing.vector().segment(sample_range.begin(), sample_range.size()).setConstant(1);
}

indices_t datasource_t::filter(tensor_size_t count, tensor_size_t condition) const
{
    indices_t indices(count);
    for (tensor_size_t sample = 0, samples = this->samples(), index = 0; sample < samples; ++sample)
    {
        if (m_testing(sample) == condition)
        {
            assert(index < indices.size());
            indices(index++) = sample;
        }
    }
    return indices;
}

void datasource_t::resize(const tensor_size_t samples, const features_t& features)
{
    this->resize(samples, features, string_t::npos);
}

void datasource_t::resize(const tensor_size_t samples, const features_t& features, const size_t target)
{
    auto       size_storage        = std::unordered_map<feature_type, tensor_size_t>{};
    const auto update_size_storage = [&](feature_type type, auto size)
    {
        const auto begin   = size_storage[type];
        const auto end     = begin + static_cast<tensor_size_t>(size);
        size_storage[type] = end;
        return std::make_pair(begin, end);
    };

    m_storage_type.resize(features.size());
    m_storage_range.resize(static_cast<tensor_size_t>(features.size()), 2);

    for (size_t i = 0, size = features.size(); i < size; ++i)
    {
        const auto& feature = features[i];

        feature_type                            type{};
        std::pair<tensor_size_t, tensor_size_t> range;
        switch (feature.type())
        {
        case feature_type::mclass:
            type  = feature_type::uint8;
            range = update_size_storage(type, feature.classes());
            break;

        case feature_type::sclass:
            type  = (feature.classes() <= (tensor_size_t(1) << 8))  ? feature_type::uint8
                  : (feature.classes() <= (tensor_size_t(1) << 16)) ? feature_type::uint16
                  : (feature.classes() <= (tensor_size_t(1) << 32)) ? feature_type::uint32
                                                                    : feature_type::uint64;
            range = update_size_storage(type, 1);
            break;

        default:
            type  = feature.type();
            range = update_size_storage(type, ::nano::size(feature.dims()));
            break;
        }

        m_storage_type[i] = type;

        const auto [begin, end]                           = range;
        m_storage_range(static_cast<tensor_size_t>(i), 0) = begin;
        m_storage_range(static_cast<tensor_size_t>(i), 1) = end;
    }

    m_testing.resize(samples);
    m_testing.zero();

    m_features = features;
    m_target   = (target < features.size()) ? static_cast<tensor_size_t>(target) : m_storage_range.size();

    m_storage_f32.resize(size_storage[feature_type::float32], samples);
    m_storage_f64.resize(size_storage[feature_type::float64], samples);
    m_storage_i08.resize(size_storage[feature_type::int8], samples);
    m_storage_i16.resize(size_storage[feature_type::int16], samples);
    m_storage_i32.resize(size_storage[feature_type::int32], samples);
    m_storage_i64.resize(size_storage[feature_type::int64], samples);
    m_storage_u08.resize(size_storage[feature_type::uint8], samples);
    m_storage_u16.resize(size_storage[feature_type::uint16], samples);
    m_storage_u32.resize(size_storage[feature_type::uint32], samples);
    m_storage_u64.resize(size_storage[feature_type::uint64], samples);

    m_storage_f32.zero();
    m_storage_f64.zero();
    m_storage_i08.zero();
    m_storage_i16.zero();
    m_storage_i32.zero();
    m_storage_i64.zero();
    m_storage_u08.zero();
    m_storage_u16.zero();
    m_storage_u32.zero();
    m_storage_u64.zero();

    m_storage_mask.resize(static_cast<tensor_size_t>(features.size()), (samples + 7) / 8);
    m_storage_mask.zero();
}

task_type datasource_t::type() const
{
    return has_target() ? m_features[static_cast<size_t>(m_target)].task() : task_type::unsupervised;
}

void datasource_t::load()
{
    const auto timer = ::nano::timer_t{};
    do_load();
    const auto elapsed = timer.elapsed();

    // NB: targets must be non-optional if a supervised task
    if (has_target())
    {
        visit_target(
            [&](const feature_t&, const auto&, const auto& mask) {
                critical(!::nano::optional(mask, samples()), "datasource[", type_id(),
                         "]: the target cannot be optional!");
            });
    }

    const auto tr_samples = train_samples();
    const auto te_samples = test_samples();

    log_info("[", type_id(), "]: loaded in <", elapsed, ">.");
    log_info("[", type_id(), "]: type=", type(), ",input features=", features(), ",samples=", samples(), "(",
             tr_samples.size(), "+", te_samples.size(), ").");
    log_info("[", type_id(), "]: target=[", (!has_target() ? "N/A" : scat(m_features[static_cast<size_t>(m_target)])),
             "].");
}

factory_t<datasource_t>& datasource_t::all()
{
    static auto manager = factory_t<datasource_t>{};
    const auto  op      = []()
    {
        manager.add<tabular_datasource_t>(
            "classify flowers from physical measurements of the sepal and petal (Fisher, 1936)", "iris",
            csvs_t{csv_t{"iris/iris.data"}.delim(",").header(false).expected(150)},
            features_t{
                feature_t{"sepal_length_cm"},
                feature_t{"sepal_width_cm"},
                feature_t{"petal_length_cm"},
                feature_t{"petal_width_cm"},
                feature_t{"class"}.sclass(3),
            },
            4);

        manager.add<tabular_datasource_t>(
            "predict the wine type from its constituents (Aeberhard, Coomans & de Vel, 1992)", "wine",
            csvs_t{csv_t{"wine/wine.data"}.delim(",").header(false).expected(178)},
            features_t{
                feature_t{"class"}.sclass(3),
                feature_t{"Alcohol"},
                feature_t{"Malic acid"},
                feature_t{"Ash"},
                feature_t{"Alcalinity of ash"},
                feature_t{"Magnesium"},
                feature_t{"Total phenols"},
                feature_t{"Flavanoids"},
                feature_t{"Nonflavanoid phenols"},
                feature_t{"Proanthocyanins"},
                feature_t{"Color intensity"},
                feature_t{"Hue"},
                feature_t{"OD280/OD315 of diluted wines"},
                feature_t{"Proline"},
            },
            0);

        manager.add<tabular_datasource_t>(
            "predict if a person makes more than 50K per year (Kohavi & Becker, 1994)", "adult",
            csvs_t{csv_t{"adult/adult.data"}.skip('|').delim(", .").header(false).expected(32561).placeholder("?"),
                   csv_t{"adult/adult.test"}
                       .skip('|')
                       .delim(", .")
                       .header(false)
                       .expected(16281)
                       .testing(make_range(0, 16281))
                       .placeholder("?")},
            features_t{
                feature_t{"age"},
                feature_t{"workclass"}.sclass(8),
                feature_t{"fnlwgt"},
                feature_t{"education"}.sclass(16),
                feature_t{"education-num"},
                feature_t{"marital-status"}.sclass(7),
                feature_t{"occupation"}.sclass(14),
                feature_t{"relationship"}.sclass(6),
                feature_t{"race"}.sclass(5),
                feature_t{"sex"}.sclass({"Female", "Male"}),
                feature_t{"capital-gain"},
                feature_t{"capital-loss"},
                feature_t{"hours-per-week"},
                feature_t{"native-country"}.sclass(41),
                feature_t{"income"}.sclass(2),
            },
            14);

        manager.add<tabular_datasource_t>(
            "predict the age of abalone from physical measurements (Waugh, 1995)", "abalone",
            csvs_t{
                csv_t{"abalone/abalone.data"}.delim(",").header(false).expected(4177).testing(make_range(3133, 4177))},
            features_t{
                feature_t{"sex"}.sclass(3),
                feature_t{"length"},
                feature_t{"diameter"},
                feature_t{"height"},
                feature_t{"whole_weight"},
                feature_t{"shucked_weight"},
                feature_t{"viscera_weight"},
                feature_t{"shell_weight"},
                feature_t{"rings"}.sclass(29),
            },
            8);

        manager.add<tabular_datasource_t>(
            "predict the burned area of the forest (Cortez & Morais, 2007)", "forest-fires",
            csvs_t{csv_t{"forest-fires/forestfires.csv"}.delim(",").header(true).expected(517)},
            features_t{feature_t{"X"}.sclass(9), feature_t{"Y"}.sclass(8), feature_t{"month"}.sclass(12),
                       feature_t{"day"}.sclass(7), feature_t{"FFMC"}, feature_t{"DMC"}, feature_t{"DC"},
                       feature_t{"ISI"}, feature_t{"temp"}, feature_t{"RH"}, feature_t{"wind"}, feature_t{"rain"},
                       feature_t{"area"}},
            12);

        manager.add<tabular_datasource_t>(
            "diagnostic breast cancer using measurements of cell nucleai (Street, Wolberg & Mangasarian, 1992)",
            "breast-cancer", csvs_t{csv_t{"breast-cancer/wdbc.data"}.delim(",").header(false).expected(569)},
            features_t{feature_t{"ID"},          feature_t{"Diagnosis"}.sclass(2),

                       feature_t{"radius1"},     feature_t{"texture1"},
                       feature_t{"perimeter1"},  feature_t{"area1"},
                       feature_t{"smoothness1"}, feature_t{"compactness1"},
                       feature_t{"concavity1"},  feature_t{"concave_points1"},
                       feature_t{"symmetry1"},   feature_t{"fractal_dimension1"},

                       feature_t{"radius2"},     feature_t{"texture2"},
                       feature_t{"perimeter2"},  feature_t{"area2"},
                       feature_t{"smoothness2"}, feature_t{"compactness2"},
                       feature_t{"concavity2"},  feature_t{"concave_points2"},
                       feature_t{"symmetry2"},   feature_t{"fractal_dimension2"},

                       feature_t{"radius3"},     feature_t{"texture3"},
                       feature_t{"perimeter3"},  feature_t{"area3"},
                       feature_t{"smoothness3"}, feature_t{"compactness3"},
                       feature_t{"concavity3"},  feature_t{"concave_points3"},
                       feature_t{"symmetry3"},   feature_t{"fractal_dimension3"}},
            1);

        manager.add<tabular_datasource_t>(
            "predict if a client has subscribed a term deposit (Moro, Laureano & Cortez, 2011)", "bank-marketing",
            csvs_t{csv_t{"bank-marketing/bank-additional-full.csv"}.delim(";\"\r").header(true).expected(41188)},
            features_t{
                feature_t{"age"},
                feature_t{"job"}.sclass(12),
                feature_t{"marital"}.sclass(4),
                feature_t{"education"}.sclass(8),
                feature_t{"default"}.sclass(3),
                feature_t{"housing"}.sclass(3),
                feature_t{"loan"}.sclass(3),
                feature_t{"contact"}.sclass(2),
                feature_t{"month"}.sclass(12),
                feature_t{"day_of_week"}.sclass(5),
                feature_t{"duration"},
                feature_t{"campaign"},
                feature_t{"pdays"},
                feature_t{"previous"},
                feature_t{"poutcome"}.sclass(3),
                feature_t{"emp.var.rate"},
                feature_t{"cons.price.idx"},
                feature_t{"cons.conf.idx"},
                feature_t{"euribor3m"},
                feature_t{"nr.employed"},
                feature_t{"y"}.sclass(2),
            },
            20);

        manager.add<mnist_datasource_t>("classify 28x28 grayscale images of hand-written digits (MNIST)");
        manager.add<cifar10_datasource_t>("classify 3x32x32 color images (CIFAR-10)");
        manager.add<cifar100c_datasource_t>("classify 3x32x32 color images (CIFAR-100 with 20 coarse labels)");
        manager.add<cifar100f_datasource_t>("classify 3x32x32 color images (CIFAR-100 with 100 fine labels)");
        manager.add<fashion_mnist_datasource_t>("classify 28x28 grayscale images of fashion articles (Fashion-MNIST)");
    };

    static std::once_flag flag;
    std::call_once(flag, op);

    return manager;
}
