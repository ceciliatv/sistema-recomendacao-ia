import pytest
from fastapi.testclient import TestClient

from repertorio.api import create_app


@pytest.fixture(scope="module")
def client(project):
    with TestClient(create_app(project)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["items"] == 72
    assert body["models"] == ["book", "movie", "music"]


def test_search(client):
    hits = client.get("/items/search", params={"q": "horror", "domain": "series"}).json()
    assert hits and all(h["domain"] == "series" for h in hits)


def test_item_and_404(client):
    assert client.get("/items/book:scifi1").json()["title"] == "Scifi Book 1"
    assert client.get("/items/book:nope").status_code == 404


def test_similar_cross_domain(client):
    recs = client.get("/items/book:scifi1/similar", params={"target_domain": "series", "k": 3}).json()
    assert len(recs) == 3 and all(r["item_id"].startswith("series:") for r in recs)


def test_recommendations_by_title(client):
    body = {"liked": ["book:Scifi Book 2"], "domain": "movie", "k": 4, "exclude_tags": ["horror"]}
    recs = client.post("/recommendations", json=body).json()
    assert len(recs) == 4 and all("horror" not in r["tags"] for r in recs)
    assert recs[0]["reasons"]


def test_recommendations_unknown_reference(client):
    response = client.post("/recommendations", json={"liked": ["book:nao existe zzz"], "domain": "book"})
    assert response.status_code == 404


def test_chat_with_heuristic_provider(client):
    body = {"message": "Quero 2 filmes sem terror", "provider": "heuristic", "profile": {"liked": ["movie:scifi0"]}}
    result = client.post("/chat", json=body).json()
    assert result["submitted"] and len(result["recommendations"]) == 2
    assert result["profile"]["liked"] == ["movie:scifi0"]
