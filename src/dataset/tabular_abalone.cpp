#include <nano/dataset/tabular_abalone.h>

using namespace nano;

abalone_dataset_t::abalone_dataset_t()
{
    features(
    {
        feature_t{"sex"}.labels({"M", "F", "I"}),
        feature_t{"length"},
        feature_t{"diameter"},
        feature_t{"height"},
        feature_t{"whole_weight"},
        feature_t{"shucked_weight"},
        feature_t{"viscera_weight"},
        feature_t{"shell_weight"},
        feature_t{"rings"}.labels({
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29"
        }),
    }, 8);

    const auto dir = scat(std::getenv("HOME"), "/libnano/datasets/abalone");
    csvs(
    {
        csv_t{dir + "/abalone.data"}.delim(",").header(false).expected(4177).testing(make_range(3133, 4177))
    });
}
