#include <fixture/enum.h>
#include <fstream>
#include <nano/core/stream.h>
#include <nano/parameter.h>
#include <utest/utest.h>

using namespace nano;

namespace
{
void check_stream(const parameter_t& param)
{
    {
        std::ofstream stream;
        UTEST_CHECK_THROW(param.write(stream), std::runtime_error);
    }
    {
        parameter_t   xparam;
        std::ifstream stream;
        UTEST_CHECK_THROW(xparam.read(stream), std::runtime_error);
    }
    string_t str;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(param.write(stream));
        str = stream.str();
    }
    {
        parameter_t        xparam;
        std::istringstream stream(str);
        UTEST_CHECK_NOTHROW(xparam.read(stream));
        UTEST_CHECK_EQUAL(param, xparam);
    }
    {
        parameter_t xparam;
        reinterpret_cast<int32_t*>(const_cast<char*>(str.data()))[0] = 42; // NOLINT
        std::istringstream stream(str);
        UTEST_CHECK_THROW(xparam.read(stream), std::runtime_error);
    }
    {
        std::ostringstream ostream;
        UTEST_CHECK_NOTHROW(::nano::write(ostream, param));

        parameter_t        xparam;
        std::istringstream istream(ostream.str());
        UTEST_CHECK_NOTHROW(::nano::read(istream, xparam));
        UTEST_CHECK_EQUAL(param, xparam);
    }
}

template <bool valid, class tvalue>
void check_value(parameter_t param, tvalue value)
{
    UTEST_CHECK_THROW(param = "", std::invalid_argument);
    UTEST_CHECK_THROW(param = "what", std::invalid_argument);
    UTEST_CHECK_THROW(param = enum_type::type1, std::runtime_error);
    UTEST_CHECK_THROW(param = std::make_tuple(value, value), std::runtime_error);

    if (valid)
    {
        UTEST_CHECK_NOTHROW(param = value);
        UTEST_CHECK_EQUAL(param.value<tvalue>(), value);

        UTEST_CHECK_NOTHROW(param = scat(value));
        UTEST_CHECK_EQUAL(param.value<tvalue>(), value);
    }
    else
    {
        const auto old_value = param.value<tvalue>();

        UTEST_CHECK_THROW(param = value, std::runtime_error);
        UTEST_CHECK_EQUAL(param.value<tvalue>(), old_value);

        UTEST_CHECK_THROW(param = scat(value), std::runtime_error);
        UTEST_CHECK_EQUAL(param.value<tvalue>(), old_value);
    }
}

template <bool valid, class tvalue>
void check_value_pair(parameter_t param, tvalue value1, tvalue value2)
{
    const auto i32pair = std::make_tuple(static_cast<int32_t>(value1), static_cast<int32_t>(value2));
    const auto i64pair = std::make_tuple(static_cast<int64_t>(value1), static_cast<int64_t>(value2));
    const auto f64pair = std::make_tuple(static_cast<scalar_t>(value1), static_cast<scalar_t>(value2));

    UTEST_CHECK_THROW(param = value1, std::runtime_error);
    UTEST_CHECK_THROW(param = value2, std::runtime_error);
    UTEST_CHECK_THROW(param = "what", std::invalid_argument);
    UTEST_CHECK_THROW(param = scat(value1), std::invalid_argument);
    UTEST_CHECK_THROW(param = scat(value2), std::invalid_argument);
    UTEST_CHECK_THROW(param = enum_type::type1, std::runtime_error);
    UTEST_CHECK_THROW(param = scat("|", value1), std::invalid_argument);
    UTEST_CHECK_THROW(param = scat(value2, "|"), std::invalid_argument);

    if (valid)
    {
        UTEST_CHECK_NOTHROW(param = i32pair);
        UTEST_CHECK_NOTHROW(param = i64pair);
        UTEST_CHECK_NOTHROW(param = f64pair);
        UTEST_CHECK_NOTHROW(param = scat(value1, ",", value2));

        std::tuple<tvalue, tvalue> values;
        UTEST_CHECK_NOTHROW(values = param.value_pair<tvalue>());
        UTEST_CHECK_EQUAL(value1, std::get<0>(values));
        UTEST_CHECK_EQUAL(value2, std::get<1>(values));
    }
    else
    {
        UTEST_CHECK_THROW(param = i32pair, std::runtime_error);
        UTEST_CHECK_THROW(param = i64pair, std::runtime_error);
        UTEST_CHECK_THROW(param = f64pair, std::runtime_error);
        UTEST_CHECK_THROW(param = scat(value1, ",", value2), std::runtime_error);
    }
}
} // namespace

UTEST_BEGIN_MODULE(test_parameter)

UTEST_CASE(monostate)
{
    auto param = parameter_t{};

    UTEST_CHECK_EQUAL(param.name(), "");
    UTEST_CHECK_EQUAL(scat(param), "=N/A|domain=[N/A]");

    UTEST_CHECK_EQUAL(param, parameter_t{});
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("", 0.0, LE, 0.5, LE, 1.0));

    UTEST_CHECK_THROW(param.value<int>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value<string_t>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value<enum_type>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value_pair<scalar_t>(), std::runtime_error);

    UTEST_CHECK_THROW(param = 1, std::runtime_error);
    UTEST_CHECK_THROW(param = "1", std::runtime_error);
    UTEST_CHECK_THROW(param = enum_type::type1, std::runtime_error);
    const auto ituple = std::make_tuple(1, 2);
    UTEST_CHECK_THROW(param = ituple, std::runtime_error);

    UTEST_CHECK_THROW(param.value<int>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value<enum_type>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value_pair<scalar_t>(), std::runtime_error);

    check_stream(param);
}

UTEST_CASE(enumeration)
{
    auto param = parameter_t::make_enum("enum", enum_type::type1);

    UTEST_CHECK_EQUAL(param.name(), "enum");
    UTEST_CHECK_EQUAL(scat(param), "enum=type1|domain=[type1,type2,type3]");
    UTEST_CHECK_NOT_EQUAL(param, parameter_t{});
    UTEST_CHECK_EQUAL(param, parameter_t::make_enum("enum", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("what", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("enum", enum_type::type2));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("", 0.0, LE, 0.5, LE, 1.0));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("enum", 0, LE, 1, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("enum", 0, LE, 1, LE, 2, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("enum", 0, LE, 1, LE, 2, LE, 10));

    UTEST_CHECK_THROW(param = 1, std::runtime_error);
    UTEST_CHECK_THROW(param = "1", std::runtime_error);
    UTEST_CHECK_THROW(param = "typeX", std::runtime_error);
    const auto ituple = std::make_tuple(1, 2);
    UTEST_CHECK_THROW(param = ituple, std::runtime_error);

    UTEST_CHECK_NOTHROW(param = enum_type::type2);
    UTEST_CHECK_THROW(param.value<int>(), std::runtime_error);
    UTEST_CHECK_EQUAL(param.value<enum_type>(), enum_type::type2);
    UTEST_CHECK_THROW(param.value_pair<scalar_t>(), std::runtime_error);

    check_stream(param);
}

UTEST_CASE(iparam)
{
    auto param = parameter_t::make_integer("iparam", 1, LE, 7, LT, 10);

    UTEST_CHECK_EQUAL(param.name(), "iparam");
    UTEST_CHECK_EQUAL(scat(param), "iparam=7|domain=[1 <= 7 < 10]");
    UTEST_CHECK_NOT_EQUAL(param, parameter_t{});
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("enum", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("iparam", 1, LE, 7, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("iparam", 1, LE, 7, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("iparam", 1, LT, 7, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("iparam", 1, LE, 6, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("iparam", 1, LE, 7, LT, 11));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("iparam", 2, LE, 7, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("xparam", 1, LE, 7, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("iparam", 1, LE, 7, LE, 7, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LE, 7, LE, 7, LT, 10));
    UTEST_CHECK_EQUAL(param, parameter_t::make_integer("iparam", 1, LE, 7, LT, 10));

    check_value<true>(param, int32_t(1));
    check_value<true>(param, int64_t(3));
    check_value<true>(param, scalar_t(6));

    check_value<false>(param, 0);
    check_value<false>(param, 10);
    check_value<false>(param, 11);

    check_stream(param);
}

UTEST_CASE(fparam)
{
    auto param = parameter_t::make_scalar("fparam", 1.0, LT, 4, LE, 10);

    UTEST_CHECK_EQUAL(param.name(), "fparam");
    UTEST_CHECK_EQUAL(scat(param), "fparam=4|domain=[1 < 4 <= 10]");
    UTEST_CHECK_NOT_EQUAL(param, parameter_t{});
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("enum", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("fparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 1, LE, 4, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 1, LE, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 1, LT, 4, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 1, LT, 4, LE, 11));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 1, LT, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 2, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("xparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 4, LE, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("fparam", 1, LT, 4, LE, 4, LE, 10));
    UTEST_CHECK_EQUAL(param, parameter_t::make_scalar("fparam", 1, LT, 4, LE, 10));

    check_value<true>(param, 1.1);
    check_value<true>(param, 2);
    check_value<true>(param, 10.0);

    check_value<false>(param, 1.0);
    check_value<false>(param, 11);

    check_stream(param);
}

UTEST_CASE(iparam2)
{
    auto param = parameter_t::make_integer_pair("iparam", 1, LE, 2, LE, 2, LT, 10);

    UTEST_CHECK_EQUAL(param.name(), "iparam");
    UTEST_CHECK_EQUAL(scat(param), "iparam=(2,2)|domain=[1 <= 2 <= 2 < 10]");
    UTEST_CHECK_NOT_EQUAL(param, parameter_t{});
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("iparam", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("iparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("iparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("iparam", 1, LE, 2, LE, 2, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("xparam", 1, LE, 2, LE, 2, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 2, LE, 2, LE, 2, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LT, 2, LE, 2, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LE, 1, LE, 2, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LE, 2, LT, 3, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LE, 2, LE, 2, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LE, 2, LE, 2, LT, 11));
    UTEST_CHECK_EQUAL(param, parameter_t::make_integer_pair("iparam", 1, LE, 2, LE, 2, LT, 10));

    check_stream(param);

    check_value_pair<true>(param, 1, 2);
    check_value_pair<true>(param, 2, 2);
    check_value_pair<true>(param, 2, 3);
    check_value_pair<true>(param, 3, 7);
    check_value_pair<true>(param, 2, 9);

    check_value_pair<false>(param, 3, 2);
    check_value_pair<false>(param, 0, 2);
    check_value_pair<false>(param, 3, 2);
    check_value_pair<false>(param, 0, 10);
    check_value_pair<false>(param, 2, 10);
}

UTEST_CASE(fparam2)
{
    auto param = parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 3, LE, 10);

    UTEST_CHECK_EQUAL(param.name(), "fparam");
    UTEST_CHECK_EQUAL(scat(param), "fparam=(2,3)|domain=[1 < 2 < 3 <= 10]");
    UTEST_CHECK_NOT_EQUAL(param, parameter_t{});
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("fparam", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("fparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("fparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("xparam", 1, LT, 2, LT, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 0, LT, 2, LT, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LE, 2, LT, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 1.5, LT, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 2, LE, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 3, LT, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 3, LE, 11));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("fparam", 1, LT, 2, LT, 3, LE, 10));
    UTEST_CHECK_EQUAL(param, parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 3, LE, 10));

    check_stream(param);

    check_value_pair<true>(param, 2, 3);
    check_value_pair<true>(param, 2, 9);
    check_value_pair<true>(param, 3, 10);

    check_value_pair<false>(param, 1, 3);
    check_value_pair<false>(param, 2, 2);
    check_value_pair<false>(param, 0, 2);
    check_value_pair<false>(param, 2, 11);
    check_value_pair<false>(param, 12, 13);
}

UTEST_CASE(string)
{
    auto param = parameter_t::make_string("sparam", "str");

    UTEST_CHECK_EQUAL(param.name(), "sparam");
    UTEST_CHECK_EQUAL(scat(param), "sparam=str|domain=[.*]");
    UTEST_CHECK_NOT_EQUAL(param, parameter_t{});
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_enum("sparam", enum_type::type1));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar("sparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer("sparam", 1, LT, 4, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_string("xparam", "str"));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_scalar_pair("fparam", 0, LT, 2, LT, 3, LE, 10));
    UTEST_CHECK_NOT_EQUAL(param, parameter_t::make_integer_pair("fparam", 1, LT, 2, LT, 3, LE, 10));
    UTEST_CHECK_EQUAL(param, parameter_t::make_string("sparam", "str"));

    UTEST_CHECK_THROW(param = 1, std::runtime_error);
    UTEST_CHECK_THROW(param = enum_type::type1, std::runtime_error);
    const auto ituple = std::make_tuple(1, 2);
    UTEST_CHECK_THROW(param = ituple, std::runtime_error);

    UTEST_CHECK_NOTHROW(param = "str2");
    UTEST_CHECK_EQUAL(param.value<string_t>(), "str2");
    UTEST_CHECK_THROW(param.value<int>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value<enum_type>(), std::runtime_error);
    UTEST_CHECK_THROW(param.value_pair<scalar_t>(), std::runtime_error);

    check_stream(param);
}

UTEST_CASE(invalid_float)
{
    UTEST_CHECK_THROW(parameter_t::make_scalar("fparam", 1, LE, 1, LT, 1), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar("fparam", 1, LE, 1, LE, 0), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar("fparam", 1, LT, 1, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar("fparam", 1, LT, 1, LT, 10), std::runtime_error);
}

UTEST_CASE(invalid_integer)
{
    UTEST_CHECK_THROW(parameter_t::make_integer("iparam", 1, LE, 1, LT, 1), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer("iparam", 1, LE, 1, LE, 0), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer("iparam", 1, LT, 1, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer("iparam", 1, LT, 1, LT, 10), std::runtime_error);
}

UTEST_CASE(invalid_float_pair)
{
    UTEST_CHECK_THROW(parameter_t::make_scalar_pair("fparam", 1, LT, 1, LT, 3, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar_pair("fparam", 2, LT, 1, LT, 3, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 2, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 3, LE, 2), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_scalar_pair("fparam", 1, LT, 2, LT, 3, LE, 1), std::runtime_error);
}

UTEST_CASE(invalid_integer_pair)
{
    UTEST_CHECK_THROW(parameter_t::make_integer_pair("iparam", 1, LT, 1, LT, 3, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer_pair("iparam", 2, LT, 1, LT, 3, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer_pair("iparam", 1, LT, 2, LT, 2, LE, 10), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer_pair("iparam", 1, LT, 2, LT, 3, LE, 2), std::runtime_error);
    UTEST_CHECK_THROW(parameter_t::make_integer_pair("iparam", 1, LT, 2, LT, 3, LE, 1), std::runtime_error);
}

UTEST_END_MODULE()
