import pytest

from repertorio.data.tags import aggregate_tags, normalize_tag


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sci-Fi", ["science-fiction"]),
        ("Science-Fiction", ["science-fiction"]),
        ("sci-fi-fantasy", ["science-fiction", "fantasy"]),
        ("Children's", ["kids"]),
        ("R&B", ["rnb"]),
        ("Hip Hop", ["hip-hop"]),
        ("dystopian", ["dystopia"]),
        ("historical-fiction", ["historical"]),
        ("MPB", ["mpb", "brazilian"]),
        ("80s", ["80s"]),
        ("melancholic", ["melancholic"]),
        ("literary-fiction", ["literary-fiction"]),
    ],
)
def test_synonyms_and_normalization(raw, expected):
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "to-read",
        "currently-reading",
        "books-i-own",
        "favorites",
        "seen live",
        "kindle",
        "(no genres listed)",
        "IMAX",
        "fiction",
        "my-favorite-books",
        "2015-reads",
        "favoritos",
    ],
)
def test_shelf_and_noise_tags_are_dropped(raw):
    assert normalize_tag(raw) == []


def test_aggregate_merges_synonyms_and_normalizes_weights():
    tags, weights = aggregate_tags([("sci-fi", 30), ("Science Fiction", 10), ("to-read", 500), ("dark", 20)])
    assert tags == ["science-fiction", "dark"]
    assert weights == [1.0, 0.5]


def test_aggregate_limits_number_of_tags():
    tags, _ = aggregate_tags([(f"tag{chr(97 + i)}x", 10 - i) for i in range(10)], max_tags=3)
    assert len(tags) == 3
