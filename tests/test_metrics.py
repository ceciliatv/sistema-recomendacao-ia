import numpy as np
import pytest

from repertorio.evals import metrics


def test_top_k_orders_and_excludes_history():
    scores = np.array([[0.9, 0.1, 0.8, 0.7, 0.2]])
    ranked = metrics.top_k(scores, 3, [np.array([0])])
    assert ranked.tolist() == [[2, 3, 4]]


def test_hit_rate_and_recall_hand_computed():
    ranked = np.array([[1, 2, 3], [4, 5, 6]])
    targets = [np.array([3]), np.array([9])]
    assert metrics.hit_rate(ranked, targets, 3) == 0.5
    assert metrics.hit_rate(ranked, targets, 2) == 0.0
    # recall com dois alvos e k=3: 1 acerto de min(3, 2) = 2
    assert metrics.recall(np.array([[7, 1, 8]]), [np.array([1, 2])], 3) == 0.5


def test_ndcg_hand_computed():
    # alvo na posição 2 (índice 1): 1/log2(3)
    assert metrics.ndcg(np.array([[5, 7, 9]]), [np.array([7])], 3) == pytest.approx(1 / np.log2(3))
    # dois alvos nas posições 1 e 3: (1 + 1/2) / (1 + 1/log2(3))
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert metrics.ndcg(np.array([[1, 0, 2]]), [np.array([1, 2])], 3) == pytest.approx(expected)


def test_coverage():
    ranked = np.array([[0, 1], [1, 2]])
    assert metrics.coverage(ranked, 10, 2) == 0.3
