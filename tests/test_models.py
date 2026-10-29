import numpy as np

from repertorio.evals.offline import evaluate
from repertorio.models.base import InteractionData, leave_one_out, new_users
from repertorio.models.itemknn import ItemKNNRecommender
from repertorio.models.popularity import PopularityRecommender
from repertorio.models.two_tower import TwoTowerConfig, TwoTowerRecommender

from .conftest import make_interactions


def _data() -> InteractionData:
    return InteractionData.from_frame("movie", make_interactions("movie", n_users=150), positive_threshold=4.0)


def test_k_core_and_temporal_order():
    data = _data()
    assert data.temporal
    assert all(len(h) >= 5 for h in data.histories)


def test_leave_one_out_hides_last_item():
    data = _data()
    split = leave_one_out(data)
    assert all(t[0] == h[-1] for t, h in zip(split.targets, data.histories, strict=True))
    assert all(len(i) == len(h) - 1 for i, h in zip(split.inputs, data.histories, strict=True))


def test_new_users_are_not_in_training():
    data = _data()
    split = new_users(data, test_fraction=0.2)
    assert len(split.train) + len(split.inputs) == len(data.histories)
    assert all(len(i) >= 1 and len(t) >= 1 for i, t in zip(split.inputs, split.targets, strict=True))


def test_personalized_models_beat_popularity_on_clustered_data():
    data = _data()
    split = new_users(data, test_fraction=0.3)
    results = {}
    models = {
        "popularity": PopularityRecommender(),
        "itemknn": ItemKNNRecommender(k=20),
        "two_tower": TwoTowerRecommender(TwoTowerConfig(dim=16, epochs=25, batch_size=64, lr=1e-2)),
    }
    for name, model in models.items():
        model.fit(split.train, data.n_items)
        results[name] = evaluate(model, split, data.n_items)["ndcg@10"]
    assert results["itemknn"] > results["popularity"]
    assert results["two_tower"] > results["popularity"]


def test_two_tower_is_inductive_and_roundtrips(tmp_path):
    data = _data()
    model = TwoTowerRecommender(TwoTowerConfig(dim=16, epochs=3, batch_size=64)).fit(data.histories, data.n_items)
    model.save(tmp_path)
    loaded = TwoTowerRecommender.load(tmp_path)
    history = [np.array([0, 1])]  # histórico de alguém que não estava no treino
    np.testing.assert_allclose(model.score(history), loaded.score(history), rtol=1e-5)
    assert loaded.item_similarity(0).shape == (data.n_items,)
