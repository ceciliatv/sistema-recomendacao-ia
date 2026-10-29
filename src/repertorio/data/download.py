"""Download idempotente das fontes públicas para `data/raw/`.

Nada aqui é versionado: MovieLens e HetRec não permitem redistribuição, então o
repositório traz só este script.
"""

from __future__ import annotations

import json
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
LASTFM_URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
GOODBOOKS_BASE = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
GOODBOOKS_FILES = ("books.csv", "ratings.csv", "book_tags.csv", "tags.csv")
TVMAZE_URL = "https://api.tvmaze.com/shows"

Log = Callable[[str], None]


def _fetch(url: str, dest: Path, log: Log) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"já existe: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"baixando {url}")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in response.iter_bytes(1 << 20):
                fh.write(chunk)
    tmp.rename(dest)


def _unzip(archive: Path, target: Path, log: Log) -> None:
    if target.exists():
        return
    log(f"extraindo {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)


def download_movielens(raw: Path, log: Log = print) -> None:
    archive = raw / "ml-latest-small.zip"
    _fetch(MOVIELENS_URL, archive, log)
    _unzip(archive, raw / "movielens", log)


def download_lastfm(raw: Path, log: Log = print) -> None:
    archive = raw / "hetrec2011-lastfm-2k.zip"
    _fetch(LASTFM_URL, archive, log)
    _unzip(archive, raw / "lastfm", log)


def download_goodbooks(raw: Path, log: Log = print) -> None:
    for name in GOODBOOKS_FILES:
        _fetch(f"{GOODBOOKS_BASE}/{name}", raw / "goodbooks" / name, log)


def download_tvmaze(raw: Path, log: Log = print, max_pages: int = 400, pause: float = 0.6) -> None:
    """Pagina o índice de séries do TVMaze até a API devolver 404 (fim da lista).

    A API pede no máximo 20 chamadas a cada 10 segundos; a pausa respeita esse limite.
    """
    folder = raw / "tvmaze"
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, headers={"User-Agent": "repertorio (portfolio project)"}) as client:
        for page in range(max_pages):
            dest = folder / f"shows_{page:03d}.json"
            if dest.exists():
                continue
            response = client.get(TVMAZE_URL, params={"page": page})
            if response.status_code == 404:
                log(f"tvmaze: fim na página {page}")
                break
            if response.status_code == 429:
                time.sleep(10)
                response = client.get(TVMAZE_URL, params={"page": page})
            response.raise_for_status()
            dest.write_text(json.dumps(response.json()))
            if page % 25 == 0:
                log(f"tvmaze: página {page}")
            time.sleep(pause)


def download_all(raw: Path, log: Log = print) -> None:
    download_movielens(raw, log)
    download_goodbooks(raw, log)
    download_lastfm(raw, log)
    download_tvmaze(raw, log)
