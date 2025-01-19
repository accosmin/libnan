#include <filesystem>
#include <fstream>
#include <nano/core/strutil.h>
#include <nano/critical.h>
#include <nano/main.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
string_t read_file(const std::filesystem::path& path)
{
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void check_logger(logger_t& logger)
{
    UTEST_CHECK_NOTHROW(logger.log(log_type::info, std::fixed, std::setprecision(6), 1.5, "message\n"));
    UTEST_CHECK_NOTHROW(logger.log(log_type::warn, std::fixed, std::setprecision(7), 1.5, "message\n"));
    UTEST_CHECK_NOTHROW(logger.log(log_type::error, std::fixed, std::setprecision(8), 1.5, "message\n"));

    auto logger1 = logger.fork("file.log");
    UTEST_CHECK_NOTHROW(logger1.log("forked file.log message"));

    auto logger2 = logger.fork("dir", "file.log");
    UTEST_CHECK_NOTHROW(logger2.log("forked dir/file message"));
}

class fixture_t
{
public:
    explicit fixture_t(std::filesystem::path path)
        : m_path(std::move(path))
    {
    }

    fixture_t(const fixture_t&)     = default;
    fixture_t(fixture_t&&) noexcept = default;

    fixture_t& operator=(const fixture_t&)     = default;
    fixture_t& operator=(fixture_t&&) noexcept = default;

    ~fixture_t() { std::filesystem::remove_all(m_path); }

    const auto& root() const { return m_path; }

private:
    // attributes
    std::filesystem::path m_path;
};
} // namespace

UTEST_BEGIN_MODULE(test_logger)

UTEST_CASE(critical)
{
    UTEST_CHECK_NOTHROW(critical(true, "message\n"));
    UTEST_CHECK_THROW(critical(false, "message\n"), std::runtime_error);
}

UTEST_CASE(main)
{
    const auto op_ok = [](int, const char*[]) { return EXIT_SUCCESS; };

    const auto op_unknown = [](int, const char*[])
    {
        throw 42;            // NOLINT(hicpp-exception-baseclass)
        return EXIT_SUCCESS; // cppcheck-suppress duplicateBreak
    };

    const auto op_exception = [](int, const char*[])
    {
        throw std::runtime_error("runtime error");
        return EXIT_SUCCESS; // cppcheck-suppress duplicateBreak
    };

    const auto  argc   = 1;
    const char* argv[] = {"main"};

    UTEST_CHECK_NOTHROW(nano::safe_main(op_ok, argc, argv));
    UTEST_CHECK_NOTHROW(nano::safe_main(op_unknown, argc, argv));
    UTEST_CHECK_NOTHROW(nano::safe_main(op_exception, argc, argv));

    UTEST_CHECK_EQUAL(nano::safe_main(op_ok, argc, argv), EXIT_SUCCESS);
    UTEST_CHECK_EQUAL(nano::safe_main(op_unknown, argc, argv), EXIT_FAILURE);
    UTEST_CHECK_EQUAL(nano::safe_main(op_exception, argc, argv), EXIT_FAILURE);
}

UTEST_CASE(null_logger)
{
    auto logger = make_null_logger();
    check_logger(logger);
}

UTEST_CASE(stdout_logger)
{
    auto logger = make_stdout_logger();
    check_logger(logger);
}

UTEST_CASE(stderr_logger)
{
    auto logger = make_stderr_logger();
    check_logger(logger);
}

UTEST_CASE(stream_logger)
{
    auto stream = std::ostringstream{};
    UTEST_CHECK_EQUAL(stream.str(), "");

    auto logger = make_stream_logger(stream);
    UTEST_CHECK_EQUAL(stream.str(), "");

    UTEST_CHECK_NOTHROW(logger.log("[date]: val=1,ret=42,prec=", std::fixed, std::setprecision(6), 0.43F));
    UTEST_CHECK_EQUAL(stream.str(), "[date]: val=1,ret=42,prec=0.430000");
}

UTEST_CASE(file_logger)
{
    const auto time    = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto fixture = fixture_t{std::filesystem::temp_directory_path() / scat(time)};

    make_stdout_logger().log(log_type::info, "using temporary directory '", fixture.root().string(), "' ...\n");
    {
        auto logger = make_file_logger((fixture.root() / "temp.log").string());
        logger.log("header\n");
        logger.log("second line: int=", std::setw(6), std::setfill('0'), 42, "\n");

        auto logger2 = logger.fork("temp2.log");
        logger2.log("data here");

        for (const auto* const fork_id : {"fold=1", "fold=2"})
        {
            auto fork_logger1 = logger.fork(fork_id, "temp1.log");
            auto fork_logger7 = logger.fork(fork_id, "temp7.log");

            fork_logger1.log(fork_id, ": value=42.7\n");
            fork_logger7.log(fork_id, ": error=10.0\n");
        }

        logger.log("third line: final result=xyz\n");
    }

    UTEST_CHECK_EQUAL(read_file(fixture.root() / "temp.log"),
                      R"XXX(header
second line: int=000042
third line: final result=xyz
)XXX");

    UTEST_CHECK_EQUAL(read_file(fixture.root() / "temp2.log"), "data here");

    UTEST_CHECK_EQUAL(read_file(fixture.root() / "fold=1" / "temp1.log"), "fold=1: value=42.7\n");
    UTEST_CHECK_EQUAL(read_file(fixture.root() / "fold=2" / "temp1.log"), "fold=2: value=42.7\n");

    UTEST_CHECK_EQUAL(read_file(fixture.root() / "fold=1" / "temp7.log"), "fold=1: error=10.0\n");
    UTEST_CHECK_EQUAL(read_file(fixture.root() / "fold=2" / "temp7.log"), "fold=2: error=10.0\n");
}

UTEST_END_MODULE()
