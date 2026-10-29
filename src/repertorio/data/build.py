"""Transforma os arquivos brutos no catálogo único e nas tabelas de interação."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from repertorio.data.schema import INTERACTION_COLUMNS, ITEM_COLUMNS, make_item_id
from repertorio.data.tags import aggregate_tags

Log = Callable[[str], None]

_YEAR_SUFFIX = re.compile(r"\s*\((\d{4})(?:[-–]\d{0,4})?\)\s*$")
_PAREN = re.compile(r"\s*\([^)]*\)")
_ARTICLE = re.compile(r"^(.*), (The|A|An|Les|La|Le|L'|Il|Das|Die|Der|El|Los|Las|O|Os|As)$")
_TAGS_HTML = re.compile(r"<[^>]+>")


def _movie_title(raw: str) -> tuple[str, int | None]:
    """"Matrix, The (1999)" -> ("The Matrix", 1999)."""
    year = None
    match = _YEAR_SUFFIX.search(raw)
    if match:
        year = int(match.group(1))
        raw = raw[: match.start()]
    raw = _PAREN.sub("", raw).strip()  # títulos alternativos entre parênteses
    article = _ARTICLE.match(raw)
    if article:
        sep = "" if article.group(2).endswith("'") else " "
        raw = f"{article.group(2)}{sep}{article.group(1)}"
    return raw, year


def _without_name_tags(tags: list[str], weights: list[float], names: list[str]) -> tuple[list, list]:
    """Remove tags que são só o nome do autor ("nelson-demille", "demille-nelson")."""
    tokens = {t for name in names for t in re.findall(r"[a-z]{3,}", name.lower())}
    pairs = [(t, w) for t, w in zip(tags, weights, strict=True) if not set(t.split("-")) <= tokens]
    return [t for t, _ in pairs], [w for _, w in pairs]


def _items_frame(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    frame["year"] = frame["year"].astype("Int64")
    frame["popularity"] = frame["popularity_raw"].rank(pct=True).astype("float32")
    return frame.drop(columns=["popularity_raw"])


# ---------------------------------------------------------------- filmes
def build_movies(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = raw / "movielens" / "ml-latest-small"
    movies = pd.read_csv(folder / "movies.csv")
    ratings = pd.read_csv(folder / "ratings.csv")
    user_tags = pd.read_csv(folder / "tags.csv")
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
    tag_counts = user_tags.groupby(["movieId", "tag"]).size()
    tags_by_movie: dict[int, list[tuple[str, float]]] = {}
    for (movie_id, tag), count in tag_counts.items():
        tags_by_movie.setdefault(int(movie_id), []).append((str(tag), 0.5 * count))

    records = []
    for row in movies.itertuples(index=False):
        if row.movieId not in stats.index or stats.at[row.movieId, "count"] < 3:
            continue
        title, year = _movie_title(row.title)
        pairs = [(g, 1.0) for g in str(row.genres).split("|")] + tags_by_movie.get(int(row.movieId), [])
        tags, weights = aggregate_tags(pairs)
        records.append(
            {
                "item_id": make_item_id("movie", row.movieId),
                "domain": "movie",
                "source_id": str(row.movieId),
                "title": title,
                "creators": [],
                "year": year,
                "tags": tags,
                "tag_weights": weights,
                "description": "",
                "popularity_raw": float(stats.at[row.movieId, "count"]),
                "rating": float(stats.at[row.movieId, "mean"]),
            }
        )
    interactions = pd.DataFrame(
        {
            "user_id": "ml" + ratings["userId"].astype(str),
            "item_id": "movie:" + ratings["movieId"].astype(str),
            "value": ratings["rating"].astype("float32"),
            "timestamp": ratings["timestamp"].astype("int64"),
        }
    )
    return _items_frame(records), interactions


# ---------------------------------------------------------------- livros
def build_books(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = raw / "goodbooks"
    books = pd.read_csv(folder / "books.csv")
    ratings = pd.read_csv(folder / "ratings.csv")
    book_tags = pd.read_csv(folder / "book_tags.csv")
    tag_names = pd.read_csv(folder / "tags.csv").set_index("tag_id")["tag_name"]
    book_tags = book_tags[book_tags["count"] > 0]
    book_tags = book_tags.assign(tag=book_tags["tag_id"].map(tag_names))
    tags_by_book: dict[int, list[tuple[str, float]]] = {}
    for gr_id, group in book_tags.groupby("goodreads_book_id"):
        tags_by_book[int(gr_id)] = list(zip(group["tag"].astype(str), group["count"].astype(float), strict=True))

    records = []
    for row in books.itertuples(index=False):
        authors = [a.strip() for a in str(row.authors).split(",") if a.strip()][:4]
        tags, weights = aggregate_tags(tags_by_book.get(int(row.goodreads_book_id), []), max_tags=20)
        tags, weights = _without_name_tags(tags, weights, authors)
        series = re.search(r"\(([^)]*#[^)]*)\)", str(row.title))
        title = _PAREN.sub("", str(row.title)).strip() or str(row.original_title)
        year = row.original_publication_year
        records.append(
            {
                "item_id": make_item_id("book", row.book_id),
                "domain": "book",
                "source_id": str(row.book_id),
                "title": title,
                "creators": authors,
                "year": int(year) if pd.notna(year) else None,
                "tags": tags,
                "tag_weights": weights,
                "description": f"Série: {series.group(1)}" if series else "",
                "popularity_raw": float(row.work_ratings_count),
                "rating": float(row.average_rating),
            }
        )
    interactions = pd.DataFrame(
        {
            "user_id": "gb" + ratings["user_id"].astype(str),
            "item_id": "book:" + ratings["book_id"].astype(str),
            "value": ratings["rating"].astype("float32"),
            "timestamp": np.zeros(len(ratings), dtype=np.int64),
        }
    )
    return _items_frame(records), interactions


# ---------------------------------------------------------------- música
def build_music(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = raw / "lastfm"
    read = {"sep": "\t", "encoding": "latin-1"}
    artists = pd.read_csv(folder / "artists.dat", **read)
    plays = pd.read_csv(folder / "user_artists.dat", **read)
    tag_names = pd.read_csv(folder / "tags.dat", **read).set_index("tagID")["tagValue"]
    tagged = pd.read_csv(folder / "user_taggedartists.dat", **read)
    tag_counts = tagged.groupby(["artistID", "tagID"]).size()
    tags_by_artist: dict[int, list[tuple[str, float]]] = {}
    for (artist_id, tag_id), count in tag_counts.items():
        tags_by_artist.setdefault(int(artist_id), []).append((str(tag_names.get(tag_id, "")), float(count)))
    listeners = plays.groupby("artistID")["userID"].nunique()

    records = []
    for row in artists.itertuples(index=False):
        if row.id not in listeners.index:
            continue
        tags, weights = aggregate_tags(tags_by_artist.get(int(row.id), []))
        tags, weights = _without_name_tags(tags, weights, [str(row.name)])
        records.append(
            {
                "item_id": make_item_id("music", row.id),
                "domain": "music",
                "source_id": str(row.id),
                "title": str(row.name),
                "creators": [],
                "year": None,
                "tags": tags,
                "tag_weights": weights,
                "description": "",
                "popularity_raw": float(listeners.at[row.id]),
                "rating": None,
            }
        )
    interactions = pd.DataFrame(
        {
            "user_id": "lfm" + plays["userID"].astype(str),
            "item_id": "music:" + plays["artistID"].astype(str),
            "value": plays["weight"].astype("float32"),
            "timestamp": np.zeros(len(plays), dtype=np.int64),
        }
    )
    return _items_frame(records), interactions


# ---------------------------------------------------------------- séries
def build_series(raw: Path, min_weight: int = 85) -> pd.DataFrame:
    records = []
    for page in sorted((raw / "tvmaze").glob("shows_*.json")):
        for show in json.loads(page.read_text()):
            weight = show.get("weight") or 0
            if weight < min_weight:
                continue
            pairs = [(g, 1.0) for g in show.get("genres") or []]
            kind = show.get("type") or ""
            if kind and kind != "Scripted":
                pairs.append((kind, 0.8))
            language = show.get("language") or ""
            if language and language != "English":
                pairs.append((language, 0.5))
            tags, weights = aggregate_tags(pairs)
            premiered = show.get("premiered") or ""
            network = (show.get("network") or show.get("webChannel") or {}).get("name")
            summary = html.unescape(_TAGS_HTML.sub("", show.get("summary") or "")).strip()
            rating = (show.get("rating") or {}).get("average")
            records.append(
                {
                    "item_id": make_item_id("series", show["id"]),
                    "domain": "series",
                    "source_id": str(show["id"]),
                    "title": show["name"],
                    "creators": [network] if network else [],
                    "year": int(premiered[:4]) if premiered[:4].isdigit() else None,
                    "tags": tags,
                    "tag_weights": weights,
                    "description": summary[:600],
                    "popularity_raw": float(weight) + (rating or 0) / 10,
                    "rating": rating / 2 if rating else None,
                }
            )
    return _items_frame(records)


# ---------------------------------------------------------------- tudo
def _drop_rare_tags(items: pd.DataFrame, min_items: int) -> pd.DataFrame:
    counts = Counter(tag for tags in items["tags"] for tag in tags)
    kept_tags, kept_weights = [], []
    for tags, weights in zip(items["tags"], items["tag_weights"], strict=True):
        pairs = [(t, w) for t, w in zip(tags, weights, strict=True) if counts[t] >= min_items]
        kept_tags.append([t for t, _ in pairs])
        kept_weights.append([w for _, w in pairs])
    items = items.assign(tags=kept_tags, tag_weights=kept_weights)
    return items[items["tags"].map(len) > 0]


def build_all(raw: Path, processed: Path, log: Log = print, min_tag_items: int = 3) -> pd.DataFrame:
    processed.mkdir(parents=True, exist_ok=True)
    log("filmes...")
    movies, movie_inter = build_movies(raw)
    log("livros...")
    books, book_inter = build_books(raw)
    log("música...")
    music, music_inter = build_music(raw)
    log("séries...")
    series = build_series(raw)

    items = pd.concat([books, movies, series, music], ignore_index=True)
    items = _drop_rare_tags(items, min_tag_items)[ITEM_COLUMNS].reset_index(drop=True)
    items.to_parquet(processed / "items.parquet", index=False)

    valid = set(items["item_id"])
    for domain, frame in (("movie", movie_inter), ("book", book_inter), ("music", music_inter)):
        frame = frame[frame["item_id"].isin(valid)][INTERACTION_COLUMNS]
        frame.to_parquet(processed / f"interactions_{domain}.parquet", index=False)
        log(f"{domain}: {frame['user_id'].nunique()} usuários, {len(frame)} interações")
    for domain, count in items["domain"].value_counts().items():
        log(f"{domain}: {count} itens")
    return items
