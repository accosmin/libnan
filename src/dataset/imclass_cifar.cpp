#include <fstream>
#include <nano/core/logger.h>
#include <nano/dataset/imclass_cifar.h>

using namespace nano;

cifar_dataset_t::cifar_dataset_t(string_t dir, string_t name, feature_t target) :
    m_dir(std::move(dir)),
    m_name(std::move(name)),
    m_target(std::move(target))
{
}

void cifar_dataset_t::file(string_t filename,
    tensor_size_t offset, tensor_size_t expected, tensor_size_t label_size, tensor_size_t label_index)
{
    m_files.emplace_back(std::move(filename), offset, expected, label_size, label_index);
}

void cifar_dataset_t::do_load()
{
    auto features = std::vector<feature_t>
    {
        feature_t("image").scalar(feature_type::uint8, make_dims(3, 32, 32)),
        m_target
    };
    resize(60000, features, 1U);

    tensor_size_t sample = 0;
    for (const auto& file : m_files)
    {
        log_info() << m_name << ": loading file <" << (m_dir + file.m_filename) << "> ...";
        critical(
            !iread(file),
            m_name, ": failed to load file <", m_dir, file.m_filename,  ">!");

        sample += file.m_expected;
        log_info() << m_name << ": loaded " << sample << " samples.";
    }

    dataset_t::testing({make_range(50000, 60000)});
}

bool cifar_dataset_t::iread(const file_t& file)
{
    std::ifstream stream(m_dir + file.m_filename);

    tensor_mem_t<int8_t, 1> label(file.m_label_size);
    tensor_mem_t<uint8_t, 3> image(3, 32, 32);

    auto expected = file.m_expected;
    for (auto sample = file.m_offset; expected > 0; -- expected, sample ++)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        if (!stream.read(reinterpret_cast<char*>(label.data()), label.size()))
        {
            return false;
        }
        set(sample, 1, static_cast<tensor_size_t>(static_cast<unsigned char>(label(file.m_label_index))));

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        if (!stream.read(reinterpret_cast<char*>(image.data()), image.size()))
        {
            return false;
        }
        set(sample, 0, image);
    }

    return expected == 0;
}

cifar10_dataset_t::cifar10_dataset_t() : cifar_dataset_t(
    scat(std::getenv("HOME"), "/libnano/datasets/cifar10/"), "CIFAR-10",
    feature_t("class").sclass(strings_t
    {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}))
{
    file("cifar-10-batches-bin/data_batch_1.bin", 0, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_2.bin", 10000, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_3.bin", 20000, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_4.bin", 30000, 10000, 1, 0);
    file("cifar-10-batches-bin/data_batch_5.bin", 40000, 10000, 1, 0);
    file("cifar-10-batches-bin/test_batch.bin", 50000, 10000, 1, 0);
}

cifar100c_dataset_t::cifar100c_dataset_t() : cifar_dataset_t(
    scat(std::getenv("HOME"), "/libnano/datasets/cifar100/"), "CIFAR-100",
    feature_t("class").sclass(strings_t
    {"aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals", "trees",
    "vehicles 1", "vehicles 2"}))
{
    file("cifar-100-binary/train.bin", 0, 50000, 2, 0);
    file("cifar-100-binary/test.bin", 50000, 10000, 2, 0);
}

cifar100f_dataset_t::cifar100f_dataset_t() : cifar_dataset_t(
    scat(std::getenv("HOME"), "/libnano/datasets/cifar100/"), "CIFAR-100",
    feature_t("class").sclass(strings_t
    {"apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup",
    "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man",
    "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"}))
{
    file("cifar-100-binary/train.bin", 0, 50000, 2, 1);
    file("cifar-100-binary/test.bin", 50000, 10000, 2, 1);
}
