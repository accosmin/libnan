#pragma once

#include <nano/dataset/tabular.h>

namespace nano
{
    ///
    /// \brief Forest fires dataset: https://archive.ics.uci.edu/ml/datasets/Forest+Fires
    ///
    class forest_fires_dataset_t final : public tabular_dataset_t
    {
    public:

        forest_fires_dataset_t();
    };
}
