#pragma once

#include <any>
#include <nano/mlearn/enums.h>
#include <nano/mlearn/stats.h>
#include <nano/tuner/space.h>

namespace nano::ml
{
///
/// \brief statistics collected while fitting a machine learning model for:
///     - a set of (train, validation) sample splits (aka folds) and
///     - a set of candidate hyper-parameter values to tune (aka trials).
///
class NANO_PUBLIC result_t
{
public:
    ///
    /// \brief default constructor
    ///
    result_t();

    ///
    /// \brief constructor
    ///
    result_t(param_spaces_t, tensor_size_t folds);

    ///
    /// \brief returns the number of folds.
    ///
    tensor_size_t folds() const { return m_values.size<1>(); }

    ///
    /// \brief returns the number of trials.
    ///
    tensor_size_t trials() const { return m_values.size<0>(); }

    ///
    /// \brief returns the hyper-parameter space to sample from.
    ///
    const param_spaces_t& param_spaces() const { return m_spaces; }

    ///
    /// \brief add the evaluation results of a hyper-parameter trial.
    ///
    void add(const tensor2d_t& params_to_try);

    ///
    /// \brief return the trial with the optimum hyper-parameter values.
    ///
    tensor_size_t optimum_trial() const;

    ///
    /// \brief returns the trial with the hyper-parameter values closest to the given ones.
    ///
    tensor_size_t closest_trial(tensor1d_cmap_t params, tensor_size_t max_trials) const;

    ///
    /// \brief set the evaluation results for the optimum hyper-parameters.
    ///
    void store(tensor2d_t errors_losses);

    ///
    /// \brief set the evaluation results for the given trial and fold.
    ///
    void store(tensor_size_t trial, tensor_size_t fold, tensor2d_t train_errors_losses, tensor2d_t valid_errors_losses,
               std::any extra = std::any{});

    ///
    /// \brief returns the hyper-parameter values for the given trial.
    ///
    tensor1d_cmap_t params(tensor_size_t trial) const;

    ///
    /// \brief returns the average value of the given trial across folds.
    ///
    scalar_t value(tensor_size_t trial, split_type = split_type::valid, value_type = value_type::errors) const;

    ///
    /// \brief returns the statistics for the optimum hyper-parameters.
    ///
    stats_t stats(ml::value_type) const;

    ///
    /// \brief returns the statistics for the given trial and fold.
    ///
    stats_t stats(tensor_size_t trial, tensor_size_t fold, split_type, value_type) const;

    ///
    /// \brief returns the model specific data stored for the given trial and fold.
    ///
    const std::any& extra(tensor_size_t trial, tensor_size_t fold) const;

private:
    using anys_t = std::vector<std::any>;

    // attributes
    param_spaces_t m_spaces; ///< hyper-parameter spaces to sample from
    tensor2d_t     m_params; ///< tried hyper-parameter values (trial, param)
    tensor5d_t     m_values; ///< results (trial, fold, train|valid, errors|losses, statistics e.g. mean|stdev)
    tensor2d_t     m_optims; ///< results at the optimum (errors|losses, statistics e.g. mean|stdev)
    anys_t         m_extras; ///< model specific data (trial, fold)
};
} // namespace nano::ml
