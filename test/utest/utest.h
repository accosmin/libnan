#pragma once

#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nano/logger.h>
#include <nano/tensor/tensor.h>
#include <string>
#include <utility>
#include <vector>

#define UTEST_STRINGIFY_(x) #x
#define UTEST_STRINGIFY(x)  UTEST_STRINGIFY_(x)

#define ERROR_COLOR "\033[35m"
#define RESET_COLOR "\033[0m"

static std::string utest_test_name;
static std::string utest_case_name;
static std::string utest_module_name;

static std::size_t              utest_n_cases    = 0;
static std::atomic<std::size_t> utest_n_checks   = {0};
static std::atomic<std::size_t> utest_n_failures = {0};

static std::mutex utest_mutex;

template <class T>
struct is_pair : std::false_type
{
};

template <class T1, class T2>
struct is_pair<std::pair<T1, T2>> : std::true_type
{
};

template <class T>
inline constexpr bool is_pair_v = is_pair<T>::value;

template <class tvalue1, class tvalue2>
static std::ostream& operator<<(std::ostream& stream, const std::pair<tvalue1, tvalue2>& pair)
{
    return stream << "{" << pair.first << "," << pair.second << "}";
}

template <size_t tindex, class... tvalues>
static std::ostream& print_tuple(std::ostream& stream, const std::tuple<tvalues...>& tuple)
{
    stream << "{" << std::get<tindex>(tuple) << "}";
    if constexpr (tindex + 1U < std::tuple_size_v<std::tuple<tvalues...>>)
    {
        print_tuple<tindex + 1U>(stream, tuple);
    }
    return stream;
}

template <class... tvalues>
static std::ostream& operator<<(std::ostream& stream, const std::tuple<tvalues...>& tuple)
{
    stream << "{";
    print_tuple<0U>(stream, tuple);
    return stream << "}";
}

template <class tvalue>
static std::ostream& operator<<(std::ostream& stream, const std::vector<tvalue>& values)
{
    stream << "{";
    for (const auto& value : values)
    {
        if constexpr (std::is_arithmetic_v<tvalue> || std::is_same_v<tvalue, std::string>)
        {
            stream << "{" << value << "}";
        }
        else if constexpr (is_pair_v<tvalue>)
        {
            stream << "{" << value.first << "," << value.second << "}";
        }
        else
        {
            stream << "{" << value << "}";
        }
    }
    return stream << "}";
}

struct utest_location_t
{
};

inline std::ostream& operator<<(std::ostream& stream, const utest_location_t&)
{
    stream << "[" << utest_module_name << "/" << utest_case_name;
    if (!utest_test_name.empty())
    {
        stream << "/" << utest_test_name;
    }
    return stream << "]";
}

struct utest_test_name_t
{
    utest_test_name_t(utest_test_name_t&&)      = default;
    utest_test_name_t(const utest_test_name_t&) = default;

    utest_test_name_t& operator=(utest_test_name_t&&)      = default;
    utest_test_name_t& operator=(const utest_test_name_t&) = default;

    explicit utest_test_name_t(std::string test_name) { utest_test_name = std::move(test_name); }

    ~utest_test_name_t() { utest_test_name.clear(); }
};

#define UTEST_NAMED_CASE(name) [[maybe_unused]] const auto utest_test_name_this = utest_test_name_t{name};

enum class exception_status
{
    none,
    expected,
    unexpected
};

template <class texception, class toperator>
static std::tuple<exception_status, nano::string_t> check_throw(const toperator& op)
{
    try
    {
        op();
        return std::make_tuple(exception_status::none, nano::string_t{});
    }
    catch (const std::exception& e)
    {
        const auto status =
            dynamic_cast<const texception*>(&e) != nullptr ? exception_status::expected : exception_status::unexpected;
        return std::make_tuple(status, nano::string_t{e.what()});
    }
    catch (...)
    {
        return std::make_tuple(exception_status::unexpected, nano::string_t{"unexpected"});
    }
}

template <class toperator>
static auto check_with_logger(const toperator& op)
{
    const auto failures = utest_n_failures.load();

    auto stream = std::ostringstream{};
    auto logger = nano::make_stream_logger(stream);

    constexpr auto returns_void = std::is_void_v<std::invoke_result_t<decltype(op), const nano::logger_t&>>;
    if constexpr (returns_void)
    {
        op(logger);
        if (failures != utest_n_failures.load())
        {
            std::cout << stream.str();
        }
    }
    else
    {
        auto result = op(logger);
        if (failures != utest_n_failures.load())
        {
            std::cout << stream.str();
        }
        return result;
    }
}

#define UTEST_BEGIN_MODULE(name)                                                                                       \
    int main(int, char*[]) /*NOLINT(hicpp-function-size,readability-function-size)*/                                   \
    {                                                                                                                  \
        try                                                                                                            \
        {                                                                                                              \
            utest_module_name = #name;

#define UTEST_CASE(name)                                                                                               \
    ++utest_n_cases;                                                                                                   \
    utest_case_name = #name;                                                                                           \
    std::cout << "running test case " << utest_location_t{} << " ..." << std::endl;

#define UTEST_END_MODULE()                                                                                             \
    if (utest_n_failures > 0)                                                                                          \
    {                                                                                                                  \
        std::cout << " failed with " << utest_n_failures << " errors in " << utest_n_checks << " check"                \
                  << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl;                                              \
        exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        std::cout << "  no errors detected in " << utest_n_checks << " check" << (utest_n_checks > 0 ? "s" : "")       \
                  << "." << std::endl;                                                                                 \
        exit(EXIT_SUCCESS);                                                                                            \
    }                                                                                                                  \
    }                                                                                                                  \
    catch (std::exception & e)                                                                                         \
    {                                                                                                                  \
        std::cout << " failed with uncaught exception <" << e.what() << ">!" << std::endl;                             \
        std::cout << " failed with " << utest_n_failures << " errors in " << utest_n_checks << " check"                \
                  << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl;                                              \
        exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        std::cout << " failed with uncaught unknown exception!" << std::endl;                                          \
        std::cout << " failed with " << utest_n_failures << " errors in " << utest_n_checks << " check"                \
                  << (utest_n_checks > 0 ? "s" : "") << "!" << std::endl;                                              \
        exit(EXIT_FAILURE);                                                                                            \
    }                                                                                                                  \
    }

#define UTEST_HANDLE_CRITICAL(critical)                                                                                \
    if (critical)                                                                                                      \
    {                                                                                                                  \
        throw std::runtime_error("critical condition failed!");                                                        \
    }
#define UTEST_HANDLE_FAILURE()                                                                                         \
    ++utest_n_failures;                                                                                                \
    std::cout << ERROR_COLOR << __FILE__ << ":" << __LINE__ << utest_location_t{} << ": "

#define UTEST_EVALUATE(check, critical)                                                                                \
    ++utest_n_checks;                                                                                                  \
    if (!(check))                                                                                                      \
    {                                                                                                                  \
        const auto _ = std::scoped_lock{utest_mutex};                                                                  \
        UTEST_HANDLE_FAILURE() << (critical ? "critical check" : "check") << " {" << UTEST_STRINGIFY(check)            \
                               << "} failed!" << RESET_COLOR << std::endl;                                             \
        UTEST_HANDLE_CRITICAL(critical)                                                                                \
    }
#define UTEST_CHECK(check)   UTEST_EVALUATE(check, false);
#define UTEST_REQUIRE(check) UTEST_EVALUATE(check, true);

#define UTEST_THROW(call, exception, critical)                                                                         \
    ++utest_n_checks;                                                                                                  \
    switch (const auto& [status, message] = check_throw<exception>([&]() { (void)(call); }); status)                   \
    {                                                                                                                  \
    case exception_status::none:                                                                                       \
        UTEST_HANDLE_FAILURE() << "call {" << UTEST_STRINGIFY(call) << "} does not throw!" << RESET_COLOR              \
                               << std::endl;                                                                           \
        UTEST_HANDLE_CRITICAL(critical);                                                                               \
        break;                                                                                                         \
    case exception_status::expected: break;                                                                            \
    case exception_status::unexpected:                                                                                 \
        UTEST_HANDLE_FAILURE() << "call {" << UTEST_STRINGIFY(call) << "} does not throw {"                            \
                               << UTEST_STRINGIFY(exception) << "}, but another exception with mesage {" << message    \
                               << "}!" << RESET_COLOR << std::endl;                                                    \
        UTEST_HANDLE_CRITICAL(critical);                                                                               \
        break;                                                                                                         \
    }
#define UTEST_CHECK_THROW(call, exception)   UTEST_THROW(call, exception, false)
#define UTEST_REQUIRE_THROW(call, exception) UTEST_THROW(call, exception, true)

#define UTEST_NOTHROW(call, critical)                                                                                  \
    ++utest_n_checks;                                                                                                  \
    switch (const auto& [status, message] = check_throw<std::exception>([&]() { (void)(call); }); status)              \
    {                                                                                                                  \
    case exception_status::none: break;                                                                                \
    case exception_status::expected:                                                                                   \
    case exception_status::unexpected:                                                                                 \
        UTEST_HANDLE_FAILURE() << "call {" << UTEST_STRINGIFY(call) << "} throws message {" << message << "}!"         \
                               << RESET_COLOR << std::endl;                                                            \
        UTEST_HANDLE_CRITICAL(critical)                                                                                \
    }
#define UTEST_CHECK_NOTHROW(call)   UTEST_NOTHROW(call, false)
#define UTEST_REQUIRE_NOTHROW(call) UTEST_NOTHROW(call, true)

#define UTEST_EVALUATE_COMPARE_OP(left, right, op, critical)                                                           \
    {                                                                                                                  \
        ++utest_n_checks;                                                                                              \
        const auto res_left  = (left);  /* NOLINT */                                                                   \
        const auto res_right = (right); /* NOLINT */                                                                   \
        const auto res_check = (res_left op res_right);                                                                \
        if (!res_check)                                                                                                \
        {                                                                                                              \
            const auto _ = std::scoped_lock{utest_mutex};                                                              \
            UTEST_HANDLE_FAILURE() << (critical ? "critical check" : "check") << " {"                                  \
                                   << UTEST_STRINGIFY(left op right) << "} failed {" << res_left << " "                \
                                   << UTEST_STRINGIFY(op) << " " << res_right << "}!" << RESET_COLOR << std::endl;     \
            UTEST_HANDLE_CRITICAL(critical)                                                                            \
        }                                                                                                              \
    }

#define UTEST_EVALUATE_NUMERIC_OP(left, right, op, critical)                                                           \
    {                                                                                                                  \
        ++utest_n_checks;                                                                                              \
        const auto res_left  = (left);  /* NOLINT */                                                                   \
        const auto res_right = (right); /* NOLINT */                                                                   \
        const auto res_check = (res_left op res_right);                                                                \
        if (!res_check)                                                                                                \
        {                                                                                                              \
            const auto _ = std::scoped_lock{utest_mutex};                                                              \
            UTEST_HANDLE_FAILURE() << (critical ? "critical check" : "check") << " {"                                  \
                                   << UTEST_STRINGIFY(left op right) << "} failed {" << res_left << " "                \
                                   << UTEST_STRINGIFY(op) << " " << res_right << "} with difference {"                 \
                                   << (res_left - res_right) << "}!" << RESET_COLOR << std::endl;                      \
            UTEST_HANDLE_CRITICAL(critical)                                                                            \
        }                                                                                                              \
    }

#define UTEST_EVALUATE_EQUAL(left, right, critical) UTEST_EVALUATE_COMPARE_OP((left), (right), ==, critical);
#define UTEST_CHECK_EQUAL(left, right)              UTEST_EVALUATE_EQUAL(left, right, false);
#define UTEST_REQUIRE_EQUAL(left, right)            UTEST_EVALUATE_EQUAL(left, right, true);

#define UTEST_EVALUATE_NOT_EQUAL(left, right, critical) UTEST_EVALUATE_COMPARE_OP(left, right, !=, critical)
#define UTEST_CHECK_NOT_EQUAL(left, right)              UTEST_EVALUATE_NOT_EQUAL(left, right, false)
#define UTEST_REQUIRE_NOT_EQUAL(left, right)            UTEST_EVALUATE_NOT_EQUAL(left, right, true)

#define UTEST_EVALUATE_LESS(left, right, critical) UTEST_EVALUATE_NUMERIC_OP(left, right, <, critical)
#define UTEST_CHECK_LESS(left, right)              UTEST_EVALUATE_LESS(left, right, false)
#define UTEST_REQUIRE_LESS(left, right)            UTEST_EVALUATE_LESS(left, right, true)

#define UTEST_EVALUATE_LESS_EQUAL(left, right, critical) UTEST_EVALUATE_NUMERIC_OP(left, right, <=, critical)
#define UTEST_CHECK_LESS_EQUAL(left, right)              UTEST_EVALUATE_LESS_EQUAL(left, right, false)
#define UTEST_REQUIRE_LESS_EQUAL(left, right)            UTEST_EVALUATE_LESS_EQUAL(left, right, true)

#define UTEST_EVALUATE_GREATER(left, right, critical) UTEST_EVALUATE_NUMERIC_OP(left, right, >, critical);
#define UTEST_CHECK_GREATER(left, right)              UTEST_EVALUATE_GREATER(left, right, false);
#define UTEST_REQUIRE_GREATER(left, right)            UTEST_EVALUATE_GREATER(left, right, true);

#define UTEST_EVALUATE_GREATER_EQUAL(left, right, critical) UTEST_EVALUATE_NUMERIC_OP(left, right, >=, critical);
#define UTEST_CHECK_GREATER_EQUAL(left, right)              UTEST_EVALUATE_GREATER_EQUAL(left, right, false);
#define UTEST_REQUIRE_GREATER_EQUAL(left, right)            UTEST_EVALUATE_GREATER_EQUAL(left, right, true);

#define UTEST_EVALUATE_CLOSE(left, right, epsilon, critical)                                                           \
    ++utest_n_checks;                                                                                                  \
    if (!::nano::close((left), (right), epsilon))                                                                      \
    {                                                                                                                  \
        const auto _ = std::scoped_lock{utest_mutex};                                                                  \
        UTEST_HANDLE_FAILURE() << (critical ? "critical check" : "check") << " {" << UTEST_STRINGIFY(left ~right)      \
                               << "} failed {" << (left) << " <" << (epsilon) << "> " << (right) << "}!"               \
                               << RESET_COLOR << std::endl;                                                            \
        UTEST_HANDLE_CRITICAL(critical);                                                                               \
    }

#define UTEST_CHECK_CLOSE(left, right, epsilon)   UTEST_EVALUATE_CLOSE(left, right, epsilon, false);
#define UTEST_REQUIRE_CLOSE(left, right, epsilon) UTEST_EVALUATE_CLOSE(left, right, epsilon, true);

#define UTEST_EVALUATE_NOT_CLOSE(left, right, epsilon, critical)                                                       \
    ++utest_n_checks;                                                                                                  \
    if (::nano::close((left), (right), epsilon))                                                                       \
    {                                                                                                                  \
        const auto _ = std::scoped_lock{utest_mutex};                                                                  \
        UTEST_HANDLE_FAILURE() << (critical ? "critical check" : "check") << " {" << UTEST_STRINGIFY(left !~right)     \
                               << "} failed {" << (left) << " <" << (epsilon) << "> " << (right) << "}!"               \
                               << RESET_COLOR << std::endl;                                                            \
        UTEST_HANDLE_CRITICAL(critical);                                                                               \
    }

#define UTEST_CHECK_NOT_CLOSE(left, right, epsilon)   UTEST_EVALUATE_NOT_CLOSE(left, right, epsilon, false);
#define UTEST_REQUIRE_NOT_CLOSE(left, right, epsilon) UTEST_EVALUATE_NOT_CLOSE(left, right, epsilon, true);
