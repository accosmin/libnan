#include <nano/critical.h>
#include <nano/tuner/space.h>

using namespace nano;

namespace
{
auto make_min(const tensor1d_t& grid_values)
{
    return (grid_values.size() < 1) ? std::numeric_limits<scalar_t>::quiet_NaN()
                                    : *std::min_element(std::begin(grid_values), std::end(grid_values));
}

auto make_max(const tensor1d_t& grid_values)
{
    return (grid_values.size() < 1) ? std::numeric_limits<scalar_t>::quiet_NaN()
                                    : *std::max_element(std::begin(grid_values), std::end(grid_values));
}
} // namespace

param_space_t::param_space_t(string_t name, param_space_t::type type_, tensor1d_t grid_values)
    : m_name(std::move(name))
    , m_type(type_)
    , m_grid_values(std::move(grid_values))
    , m_min(make_min(m_grid_values))
    , m_max(make_max(m_grid_values))
{
    critical(m_grid_values.size() >= 2, "parameter space [", m_name, "]: at least two grid values must be given!");

    critical(std::is_sorted(std::begin(m_grid_values), std::end(m_grid_values)), "parameter space [", m_name,
             "]: the grid values must be sorted!");

    critical(std::unique(std::begin(m_grid_values), std::end(m_grid_values)) == std::end(m_grid_values),
             "parameter space [", m_name, "]: the grid values must be distinct!");

    critical(m_type != type::log10 || *std::min_element(std::begin(m_grid_values), std::end(m_grid_values)) >=
                                          std::numeric_limits<scalar_t>::epsilon(),
             "parameter space [", m_name,
             "]: the grid values must be strictly positive if using the logarithmic scale!");
}

scalar_t param_space_t::to_surrogate(const scalar_t value) const
{
    critical(value >= m_min && value <= m_max, "parameter space [", m_name, "]: cannot map value (", value,
             ") outside the parameter grid range [", m_min, ",", m_max, "]!");

    switch (m_type)
    {
    case type::linear: return (value - m_min) / (m_max - m_min);

    default: return std::log10(value);
    }
}

scalar_t param_space_t::from_surrogate(const scalar_t value) const
{
    switch (m_type)
    {
    case type::linear: return std::clamp(m_min + value * (m_max - m_min), m_min, m_max);

    default: return std::clamp(std::pow(10.0, value), m_min, m_max);
    }
}

scalar_t param_space_t::closest_grid_value_from_surrogate(const scalar_t value) const
{
    return m_grid_values(closest_grid_point_from_surrogate(value));
}

tensor_size_t param_space_t::closest_grid_point_from_surrogate(const scalar_t value) const
{
    auto min_distance  = std::numeric_limits<scalar_t>::max();
    auto closest_point = tensor_size_t{0};

    for (tensor_size_t point = 0; point < m_grid_values.size(); ++point)
    {
        const auto distance = std::fabs(value - to_surrogate(m_grid_values(point)));
        if (distance < min_distance)
        {
            min_distance  = distance;
            closest_point = point;
        }
    }

    return closest_point;
}
