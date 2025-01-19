#include <fixture/configurable.h>
#include <fixture/enum.h>

using namespace nano;

namespace
{
auto to_string(const configurable_t& configurable)
{
    std::ostringstream stream;
    UTEST_REQUIRE_NOTHROW(configurable.write(stream));
    UTEST_REQUIRE(stream);
    return stream.str();
}
} // namespace

UTEST_BEGIN_MODULE(test_configurable)

UTEST_CASE(string)
{
    for (const auto& string : {std::string{}, std::string("stream strings")})
    {
        std::ostringstream ostream;
        UTEST_REQUIRE_NOTHROW(::nano::write(ostream, string));
        UTEST_REQUIRE(ostream);

        const auto ostring = ostream.str();
        UTEST_CHECK_EQUAL(ostring.size(), string.size() + 4);

        std::string        istring;
        std::istringstream istream(ostring);
        UTEST_REQUIRE(istream);
        UTEST_REQUIRE_NOTHROW(::nano::read(istream, istring));
        UTEST_REQUIRE(istream);

        UTEST_CHECK_EQUAL(string, istring);

        std::string   ifstring;
        std::ifstream ifstream;
        UTEST_REQUIRE(ifstream);
        UTEST_REQUIRE_NOTHROW(::nano::read(ifstream, ifstring));
        UTEST_REQUIRE(!ifstream);
    }
}

UTEST_CASE(vector)
{
    const auto vector = std::vector<int32_t>{2, 3};

    std::ostringstream ostream;
    UTEST_REQUIRE_NOTHROW(::nano::write(ostream, vector));
    UTEST_REQUIRE(ostream);

    const auto ostring = ostream.str();
    UTEST_CHECK_EQUAL(ostring.size(), 4U * vector.size() + 8U);

    std::vector<int32_t> ivector;
    std::istringstream   istream(ostring);
    UTEST_REQUIRE(istream);
    UTEST_REQUIRE_NOTHROW(::nano::read(istream, ivector));
    UTEST_REQUIRE(istream);

    UTEST_CHECK_EQUAL(vector, ivector);

    {
        std::ifstream ifstream;
        UTEST_REQUIRE(ifstream);
        UTEST_REQUIRE_NOTHROW(::nano::read(ifstream, ivector));
        UTEST_REQUIRE(!ifstream);
    }
    {
        std::ofstream ofstream;
        UTEST_REQUIRE(ofstream);
        UTEST_REQUIRE_NOTHROW(::nano::write(ofstream, ivector));
        UTEST_REQUIRE(!ofstream);
    }
}

UTEST_CASE(configurable_default)
{
    const auto configurable = configurable_t{};
    UTEST_CHECK_EQUAL(configurable.major_version(), ::nano::major_version);
    UTEST_CHECK_EQUAL(configurable.minor_version(), ::nano::minor_version);
    UTEST_CHECK_EQUAL(configurable.patch_version(), ::nano::patch_version);
}

UTEST_CASE(configurable_read_const)
{
    auto configurable = configurable_t{};

    auto str = to_string(configurable);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(configurable.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(configurable.major_version(), ::nano::major_version);
    UTEST_CHECK_EQUAL(configurable.minor_version(), ::nano::minor_version);
    UTEST_CHECK_EQUAL(configurable.patch_version(), ::nano::patch_version);
}

UTEST_CASE(configurable_read_major)
{
    auto configurable = configurable_t{};

    auto str = to_string(configurable);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = ::nano::major_version - 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(configurable.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(configurable.major_version(), ::nano::major_version - 1);
    UTEST_CHECK_EQUAL(configurable.minor_version(), ::nano::minor_version - 0);
    UTEST_CHECK_EQUAL(configurable.patch_version(), ::nano::patch_version - 0);
}

UTEST_CASE(configurable_read_minor)
{
    auto configurable = configurable_t{};

    auto str = to_string(configurable);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[1] = ::nano::minor_version - 2; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(configurable.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(configurable.major_version(), ::nano::major_version - 0);
    UTEST_CHECK_EQUAL(configurable.minor_version(), ::nano::minor_version - 2);
    UTEST_CHECK_EQUAL(configurable.patch_version(), ::nano::patch_version - 0);
}

UTEST_CASE(configurable_read_patch)
{
    auto configurable = configurable_t{};

    auto str = to_string(configurable);
    UTEST_REQUIRE_EQUAL(str.size(), size_t(3 * 4 + 8));
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = ::nano::patch_version - 3; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_NOTHROW(configurable.read(stream));
    UTEST_REQUIRE(stream);
    UTEST_REQUIRE_EQUAL(static_cast<size_t>(stream.tellg()), str.size());

    UTEST_CHECK_EQUAL(configurable.major_version(), ::nano::major_version - 0);
    UTEST_CHECK_EQUAL(configurable.minor_version(), ::nano::minor_version - 0);
    UTEST_CHECK_EQUAL(configurable.patch_version(), ::nano::patch_version - 3);
}

UTEST_CASE(configurable_write_fail)
{
    const auto configurable = configurable_t{};

    std::ofstream stream;
    UTEST_CHECK_THROW(configurable.write(stream), std::runtime_error);
}

UTEST_CASE(configurable_read_fail_major)
{
    auto configurable = configurable_t{};

    auto str                                                     = to_string(configurable);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = ::nano::major_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(configurable.read(stream), std::runtime_error);
}

UTEST_CASE(configurable_read_fail_minor)
{
    auto configurable = configurable_t{};

    auto str                                                     = to_string(configurable);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[1] = ::nano::minor_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(configurable.read(stream), std::runtime_error);
}

UTEST_CASE(configurable_read_fail_patch)
{
    auto configurable = configurable_t{};

    auto str                                                     = to_string(configurable);
    reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[2] = ::nano::patch_version + 1; // NOLINT

    std::istringstream stream(str);
    UTEST_REQUIRE_THROW(configurable.read(stream), std::runtime_error);
}

UTEST_CASE(no_parameters)
{
    const auto check_params = [](const configurable_t& configurable)
    { UTEST_CHECK(configurable.parameters().empty()); };

    auto configurable = configurable_t{};
    check_params(configurable);

    const auto* const pname = "nonexistent_param_name";
    const auto        sname = string_t{"unknown_param_name"};

    UTEST_CHECK_THROW(configurable.parameter(pname), std::runtime_error);
    UTEST_CHECK_THROW(configurable.parameter(sname), std::runtime_error);
    UTEST_CHECK_THROW(const_cast<const configurable_t&>(configurable).parameter(pname), // NOLINT
                      std::runtime_error);
    UTEST_CHECK_THROW(const_cast<const configurable_t&>(configurable).parameter(sname), // NOLINT
                      std::runtime_error);

    UTEST_CHECK(configurable.parameter_if(pname) == nullptr);
    UTEST_CHECK(configurable.parameter_if(sname) == nullptr);
    UTEST_CHECK(const_cast<const configurable_t&>(configurable).parameter_if(pname) == nullptr); // NOLINT
    UTEST_CHECK(const_cast<const configurable_t&>(configurable).parameter_if(sname) == nullptr); // NOLINT

    check_params(check_stream(configurable));
}

UTEST_CASE(parameters)
{
    const auto eparam = parameter_t::make_enum("eparam", enum_type::type3);
    const auto iparam = parameter_t::make_integer("iparam", 1, LE, 5, LE, 9);
    const auto fparam = parameter_t::make_scalar_pair("fparam", 1.0, LT, 2.0, LE, 2.0, LT, 5.0);

    const auto check_params = [&](const configurable_t& configurable)
    {
        UTEST_CHECK_EQUAL(configurable.parameters().size(), 3U);

        UTEST_CHECK_EQUAL(configurable.parameter("eparam"), eparam);
        UTEST_CHECK_EQUAL(configurable.parameter("iparam"), iparam);
        UTEST_CHECK_EQUAL(configurable.parameter("fparam"), fparam);
    };

    auto configurable = configurable_t{};
    UTEST_CHECK_NOTHROW(configurable.register_parameter(eparam));
    UTEST_CHECK_NOTHROW(configurable.register_parameter(iparam));
    UTEST_CHECK_NOTHROW(configurable.register_parameter(fparam));

    check_params(configurable);
    check_params(check_stream(configurable));

    UTEST_CHECK_THROW(configurable.register_parameter(eparam), std::runtime_error);
    UTEST_CHECK_THROW(configurable.register_parameter(iparam), std::runtime_error);
    UTEST_CHECK_THROW(configurable.register_parameter(fparam), std::runtime_error);

    check_params(configurable);
    check_params(check_stream(configurable));
}

UTEST_END_MODULE()
