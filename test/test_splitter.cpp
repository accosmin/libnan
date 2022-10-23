#include <nano/splitter.h>
#include <utest/utest.h>

using namespace nano;

static auto make_samples(const tensor_size_t samples)
{
    return arange(samples, 2 * samples);
}

static auto make_splits(const tensor_size_t samples, const tensor_size_t folds, const uint64_t seed, const string_t& id)
{
    auto splitter = splitter_t::all().get(id);
    UTEST_REQUIRE(splitter);

    splitter->parameter("splitter::seed")  = seed;
    splitter->parameter("splitter::folds") = folds;

    return splitter->split(make_samples(samples));
}

static void check_split(const indices_t& train, const indices_t& valid, const tensor_size_t samples)
{
    UTEST_CHECK_EQUAL(train.size() + valid.size(), samples);

    // within expected range
    UTEST_CHECK_LESS(train.max(), 2 * samples);
    UTEST_CHECK_LESS(valid.max(), 2 * samples);

    UTEST_CHECK_GREATER_EQUAL(train.min(), samples);
    UTEST_CHECK_GREATER_EQUAL(valid.min(), samples);

    // sorted splits
    UTEST_CHECK(std::is_sorted(begin(train), end(train)));
    UTEST_CHECK(std::is_sorted(begin(valid), end(valid)));

    // unique sample indices per split
    UTEST_CHECK(std::adjacent_find(begin(train), end(train)) == end(train));
    UTEST_CHECK(std::adjacent_find(begin(valid), end(valid)) == end(valid));

    // disjoint splits
    for (tensor_size_t sample = samples; sample < 2 * samples; ++sample)
    {
        const auto* const it = std::find(begin(train), end(train), sample);
        const auto* const iv = std::find(begin(valid), end(valid), sample);

        UTEST_CHECK((it == end(train) && iv != end(valid)) || (it != end(train) && iv == end(valid)));
    }
}

UTEST_BEGIN_MODULE(test_splitter)

UTEST_CASE(kfold)
{
    const auto folds   = tensor_size_t{5};
    const auto samples = tensor_size_t{25};

    auto all_valids = indices_t{samples};

    for (const auto seed : {42U, 11U, 122U})
    {
        const auto splits = make_splits(samples, folds, seed, "k-fold");
        UTEST_CHECK_EQUAL(splits.size(), static_cast<size_t>(folds));

        tensor_size_t size = 0;
        for (const auto& [train, valid] : splits)
        {
            UTEST_CHECK_EQUAL(train.size(), 20);
            UTEST_CHECK_EQUAL(valid.size(), 5);

            check_split(train, valid, samples);

            all_valids.vector().segment(size, valid.size()) = valid.vector();
            size += valid.size();
        }

        // validation splits should be disjoint and concatenate to make the full samples
        std::sort(begin(all_valids), end(all_valids));
        UTEST_CHECK_EQUAL(all_valids, make_samples(samples));
    }
}

UTEST_CASE(random)
{
    const auto folds   = tensor_size_t{5};
    const auto samples = tensor_size_t{30};

    for (const auto seed : {42U, 11U, 122U})
    {
        const auto splits = make_splits(samples, folds, seed, "random");
        UTEST_CHECK_EQUAL(splits.size(), static_cast<size_t>(folds));

        for (const auto& [train, valid] : splits)
        {
            UTEST_CHECK_EQUAL(train.size(), 24);
            UTEST_CHECK_EQUAL(valid.size(), 6);

            check_split(train, valid, samples);
        }
    }
}

UTEST_CASE(consistent)
{
    const auto folds   = tensor_size_t{5};
    const auto samples = tensor_size_t{21};

    for (const auto& id : splitter_t::all().ids())
    {
        const auto splits10  = make_splits(samples, folds, 10U, id);
        const auto splits11  = make_splits(samples, folds, 11U, id);
        const auto splits42a = make_splits(samples, folds, 42U, id);
        const auto splits42b = make_splits(samples, folds, 42U, id);

        UTEST_REQUIRE_EQUAL(splits10.size(), folds);
        UTEST_REQUIRE_EQUAL(splits11.size(), folds);
        UTEST_REQUIRE_EQUAL(splits42a.size(), folds);
        UTEST_REQUIRE_EQUAL(splits42b.size(), folds);

        for (size_t fold = 0U; fold < splits10.size(); ++fold)
        {
            UTEST_CHECK_EQUAL(splits42a[fold].first, splits42b[fold].first);
            UTEST_CHECK_EQUAL(splits42a[fold].second, splits42b[fold].second);

            UTEST_CHECK_NOT_EQUAL(splits10[fold].first, splits11[fold].first);
            UTEST_CHECK_NOT_EQUAL(splits10[fold].first, splits42a[fold].first);
            UTEST_CHECK_NOT_EQUAL(splits10[fold].first, splits42b[fold].first);
            UTEST_CHECK_NOT_EQUAL(splits11[fold].first, splits42a[fold].first);
            UTEST_CHECK_NOT_EQUAL(splits11[fold].first, splits42b[fold].first);

            UTEST_CHECK_NOT_EQUAL(splits10[fold].second, splits11[fold].second);
            UTEST_CHECK_NOT_EQUAL(splits10[fold].second, splits42a[fold].second);
            UTEST_CHECK_NOT_EQUAL(splits10[fold].second, splits42b[fold].second);
            UTEST_CHECK_NOT_EQUAL(splits11[fold].second, splits42a[fold].second);
            UTEST_CHECK_NOT_EQUAL(splits11[fold].second, splits42b[fold].second);
        }
    }
}

UTEST_END_MODULE()