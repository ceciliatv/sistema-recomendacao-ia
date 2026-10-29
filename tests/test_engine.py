from repertorio.data.schema import domain_of
from repertorio.engine import collaborative_weight
from repertorio.profile import TasteProfile, resolve_reference
from repertorio.retrieval.content import Filters


def test_cross_domain_from_books_to_series(engine):
    profile = TasteProfile(liked=["book:scifi0", "book:scifi1", "book:scifi2"])
    recs = engine.recommend(profile, "series", k=5)
    assert recs and all(domain_of(r.item_id) == "series" for r in recs)
    assert all(r.item_id.startswith("series:scifi") for r in recs[:3])
    assert "tags em comum" in recs[0].reasons[0]
    assert "livro" in recs[0].reasons[0]  # explica que veio de outro domínio


def test_collaborative_signal_used_inside_domain(engine):
    profile = TasteProfile(liked=["movie:horror0", "movie:horror1"])
    recs = engine.recommend(profile, "movie", k=4)
    assert "collaborative" in recs[0].signals
    assert all(r.item_id.startswith("movie:horror") for r in recs)
    assert any("costuma curtir" in reason for r in recs for reason in r.reasons)


def test_empty_profile_falls_back_to_popularity(engine):
    recs = engine.recommend(TasteProfile(), "book", k=3)
    pops = [r.signals["popularity"] for r in recs]
    assert pops == sorted(pops, reverse=True)
    assert "populares" in recs[0].reasons[0]


def test_preferred_tags_restrict_cold_start(engine):
    recs = engine.recommend(TasteProfile(preferred_tags=["romance"]), "movie", k=5)
    assert recs and all("romance" in engine.content.tag_sets[engine.catalog.row(r.item_id)] for r in recs)


def test_filters_and_consumed_items_are_respected(engine):
    profile = TasteProfile(liked=["movie:scifi0"], avoided_tags=["horror"])
    recs = engine.recommend(profile, "movie", k=20, filters=Filters(year_min=2000, exclude_tags=["romance"]))
    assert recs
    for r in recs:
        item = engine.catalog.get(r.item_id)
        assert r.item_id != "movie:scifi0"
        assert item["year"] >= 2000
        assert not {"horror", "romance"} & set(item["tags"])


def test_collaborative_weight_grows_with_history():
    assert collaborative_weight(0) == 0
    assert collaborative_weight(1) < collaborative_weight(3) < collaborative_weight(30) < 1


def test_similar_across_domains(engine):
    recs = engine.similar("book:horror0", target_domain="music", k=3)
    assert all(r.item_id.startswith("music:") for r in recs)
    assert recs[0].item_id.startswith("music:horror")


def test_resolve_reference(catalog):
    assert resolve_reference(catalog, "book:scifi1") == "book:scifi1"
    assert resolve_reference(catalog, "movie:Scifi Movie 2") == "movie:scifi2"
    assert resolve_reference(catalog, "Romance Series 3") == "series:romance3"
    assert resolve_reference(catalog, "book:titulo que nao existe xyz") is None
