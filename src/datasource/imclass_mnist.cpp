#include <fstream>
#include <nano/datasource/imclass_mnist.h>

using namespace nano;

base_mnist_datasource_t::base_mnist_datasource_t(string_t id, string_t dir, feature_t target)
    : datasource_t(std::move(id))
    , m_dir(std::move(dir))
    , m_target(std::move(target))
{
}

string_t base_mnist_datasource_t::make_full_path(const string_t& path) const
{
    const auto basedir = parameter("datasource::basedir").value<string_t>();
    return basedir + "/" + path;
}

features_t base_mnist_datasource_t::make_features() const
{
    return {feature_t("image").scalar(feature_type::uint8, make_dims(1, 28, 28)), m_target};
}

void base_mnist_datasource_t::do_load()
{
    const auto parts = {
        std::make_tuple(make_full_path(m_dir + "/train-images-idx3-ubyte"),
                        make_full_path(m_dir + "/train-labels-idx1-ubyte"), tensor_size_t(0), tensor_size_t(60000)),
        std::make_tuple(make_full_path(m_dir + "/t10k-images-idx3-ubyte"),
                        make_full_path(m_dir + "/t10k-labels-idx1-ubyte"), tensor_size_t(60000), tensor_size_t(10000))};

    resize(70000, make_features(), 1U);

    tensor_size_t sample = 0;
    for (const auto& [ifile, tfile, offset, expected] : parts)
    {
        log_info("[", type_id(), "]: loading file <", ifile, ">...");
        critical(iread(ifile, offset, expected), "datasource[", type_id(), "]: failed to load file <", ifile, ">!");

        log_info("[", type_id(), "]: loading file <", tfile, ">...");
        critical(tread(tfile, offset, expected), "datasource[", type_id(), "]: failed to load file <", tfile, ">!");

        sample += expected;
        log_info("[", type_id(), "]: loaded ", sample, " samples.");
    }

    datasource_t::testing({make_range(60000, 70000)});
}

bool base_mnist_datasource_t::iread(const string_t& path, tensor_size_t sample, tensor_size_t expected)
{
    char                     buffer[16];
    tensor_mem_t<uint8_t, 3> image(1, 28, 28);

    std::ifstream stream(path);
    if (!stream.read(buffer, 16))
    {
        return false;
    }

    for (; expected > 0; --expected, ++sample)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        if (!stream.read(reinterpret_cast<char*>(image.data()),
                         image.size())) // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        {
            return false;
        }
        set(sample, 0, image);
    }

    return expected == 0;
}

bool base_mnist_datasource_t::tread(const string_t& path, tensor_size_t sample, tensor_size_t expected)
{
    std::ifstream stream(path);

    if (char buffer[8]; !stream.read(buffer, 8))
    {
        return false;
    }

    eigen_vector_t<char> labels(expected);
    if (!stream.read(labels.data(), labels.size()))
    {
        return false;
    }

    for (auto i = 0; expected > 0; --expected, ++sample, ++i)
    {
        set(sample, 1, static_cast<tensor_size_t>(static_cast<unsigned char>(labels(i))));
    }

    return true;
}

mnist_datasource_t::mnist_datasource_t()
    : base_mnist_datasource_t("mnist", "mnist",
                              feature_t("digit").sclass(strings_t{"digit0", "digit1", "digit2", "digit3", "digit4",
                                                                  "digit5", "digit6", "digit7", "digit8", "digit9"}))
{
}

rdatasource_t mnist_datasource_t::clone() const
{
    return std::make_unique<mnist_datasource_t>(*this);
}

fashion_mnist_datasource_t::fashion_mnist_datasource_t()
    : base_mnist_datasource_t(
          "fashion-mnist", "fashion-mnist",
          feature_t("article").sclass(strings_t{"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
                                                "Shirt", "Sneaker", "Bag", "Ankle boot"}))
{
}

rdatasource_t fashion_mnist_datasource_t::clone() const
{
    return std::make_unique<fashion_mnist_datasource_t>(*this);
}
