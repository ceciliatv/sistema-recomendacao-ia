"""Normalização de tags entre domínios.

Cada fonte tem seu vocabulário: gêneros do MovieLens ("Sci-Fi"), prateleiras do
Goodreads ("to-read", "sci-fi-fantasy"), tags do Last.fm ("seen live", "melancholic")
e gêneros do TVMaze ("Science-Fiction"). Aqui tudo vira um vocabulário só, e as
tags de prateleira, que não dizem nada sobre o conteúdo, são descartadas.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

# tag normalizada -> tags canônicas (uma tag pode virar várias: "sci-fi-fantasy")
SYNONYMS: dict[str, list[str]] = {
    # ficção científica e fantasia
    "sci-fi": ["science-fiction"],
    "scifi": ["science-fiction"],
    "sf": ["science-fiction"],
    "science-fiction": ["science-fiction"],
    "sci-fi-fantasy": ["science-fiction", "fantasy"],
    "science-fiction-fantasy": ["science-fiction", "fantasy"],
    "fantasy-sci-fi": ["science-fiction", "fantasy"],
    "space-opera": ["space", "science-fiction"],
    "cyberpunk": ["cyberpunk", "science-fiction"],
    "epic-fantasy": ["fantasy", "epic"],
    "high-fantasy": ["fantasy", "epic"],
    "urban-fantasy": ["fantasy", "urban-fantasy"],
    "dark-fantasy": ["fantasy", "dark"],
    "dystopian": ["dystopia"],
    "dystopia": ["dystopia"],
    "postapocalyptic": ["post-apocalyptic"],
    "apocalyptic": ["post-apocalyptic"],
    "vampire": ["vampires"],
    "zombie": ["zombies"],
    "magical": ["magic"],
    "supernatural": ["supernatural"],
    "paranormal": ["supernatural"],
    # suspense
    "mysteries": ["mystery"],
    "detective": ["mystery"],
    "whodunit": ["mystery"],
    "mystery-thriller": ["mystery", "thriller"],
    "thrillers": ["thriller"],
    "suspense": ["thriller"],
    "psychological-thriller": ["psychological", "thriller"],
    "crime-fiction": ["crime"],
    "film-noir": ["noir"],
    "espionage": ["spy"],
    "spies": ["spy"],
    "scary": ["horror"],
    # tom e humor
    "humor": ["comedy"],
    "humour": ["comedy"],
    "funny": ["comedy"],
    "comedic": ["comedy"],
    "rom-com": ["romance", "comedy"],
    "romantic-comedy": ["romance", "comedy"],
    "romantic": ["romance"],
    "love-story": ["romance"],
    "darkness": ["dark"],
    "melancholy": ["melancholic"],
    "sad": ["melancholic"],
    "chillout": ["chill"],
    "chill-out": ["chill"],
    "relaxing": ["chill"],
    "mellow": ["chill"],
    "action-adventure": ["action", "adventure"],
    "philosophical": ["philosophy"],
    "magical-realism": ["magical-realism"],
    # público e formato
    "ya": ["young-adult"],
    "teen": ["young-adult"],
    "young-adult-fiction": ["young-adult"],
    "children": ["kids"],
    "childrens": ["kids"],
    "childrens-books": ["kids"],
    "picture-books": ["kids"],
    "children-s": ["kids"],
    "animated": ["animation"],
    "anime": ["anime", "animation"],
    "graphic-novel": ["comics"],
    "graphic-novels": ["comics"],
    "comic-book": ["comics"],
    "comic": ["comics"],
    "superheroes": ["superhero"],
    "musicals": ["musical"],
    "poems": ["poetry"],
    "short-story": ["short-stories"],
    # não ficção
    "non-fiction": ["nonfiction"],
    "nonfiction": ["nonfiction"],
    "documentaries": ["documentary", "nonfiction"],
    "documentary": ["documentary", "nonfiction"],
    "biographies": ["biography", "nonfiction"],
    "biography": ["biography", "nonfiction"],
    "memoir": ["biography", "nonfiction"],
    "memoirs": ["biography", "nonfiction"],
    "autobiography": ["biography", "nonfiction"],
    "true-crime": ["true-crime", "crime", "nonfiction"],
    "historical-fiction": ["historical"],
    "history-fiction": ["historical"],
    "ww2": ["war"],
    "wwii": ["war"],
    "world-war-ii": ["war"],
    "westerns": ["western"],
    "classics": ["classic"],
    "classic-literature": ["classic"],
    # outras línguas
    "misterio": ["mystery"],
    "fantasia": ["fantasy"],
    "terror": ["horror"],
    "suspenso": ["thriller"],
    "ficcion": [],
    "ciencia-ficcion": ["science-fiction"],
    "ficcao-cientifica": ["science-fiction"],
    "romantico": ["romance"],
    "romantica": ["romance"],
    "comedia": ["comedy"],
    "engracado": ["comedy"],
    "distopia": ["dystopia"],
    "distopico": ["dystopia"],
    "distopica": ["dystopia"],
    "policial": ["crime"],
    "misterios": ["mystery"],
    "documentario": ["documentary", "nonfiction"],
    "biografia": ["biography", "nonfiction"],
    "animacao": ["animation"],
    "aventura": ["adventure"],
    "acao": ["action"],
    "infantil": ["kids"],
    "classico": ["classic"],
    "classicos": ["classic"],
    "historico": ["historical"],
    "guerra": ["war"],
    "melancolico": ["melancholic"],
    "melancolica": ["melancholic"],
    "triste": ["melancholic"],
    "sombrio": ["dark"],
    "sombria": ["dark"],
    "sobrenatural": ["supernatural"],
    "magia": ["magic"],
    "espacial": ["space"],
    "eletronica": ["electronic"],
    "classica": ["classical"],
    # música
    "hiphop": ["hip-hop"],
    "hip-hop-rap": ["hip-hop"],
    "rap": ["hip-hop"],
    "r-and-b": ["rnb"],
    "r-b": ["rnb"],
    "rhythm-and-blues": ["rnb"],
    "electronica": ["electronic"],
    "electro": ["electronic"],
    "alternative-rock": ["alternative", "rock"],
    "alt-rock": ["alternative", "rock"],
    "indie-rock": ["indie", "rock"],
    "indie-pop": ["indie", "pop"],
    "pop-rock": ["pop", "rock"],
    "heavy-metal": ["metal"],
    "punk-rock": ["punk", "rock"],
    "classical-music": ["classical"],
    "female-vocalist": ["female-vocalists"],
    "bossa-nova": ["bossa-nova", "brazilian"],
    "brasil": ["brazilian"],
    "brazil": ["brazilian"],
    "brasileiro": ["brazilian"],
    "chicklit": ["chick-lit"],
    "mpb": ["mpb", "brazilian"],
}

# partes que denunciam tag de prateleira, opinião ou posse, não de conteúdo
_SHELF_PARTS = {
    "read", "reading", "reads", "reread", "re-read", "to-read", "tbr", "dnf", "unfinished", "abandoned",
    "own", "owned", "have", "bought", "buy", "borrowed", "library", "wishlist", "wish", "want",
    "kindle", "ebook", "ebooks", "e-book", "audiobook", "audiobooks", "audio", "nook", "ibooks",
    "favorite", "favorites", "favourite", "favourites", "fave", "faves", "fav", "favs",
    "default", "shelf", "shelves", "book", "books", "novel", "novels", "my", "mine", "i",
    "series", "seen", "live", "check", "spotify", "lastfm", "albums", "listened",
    "awesome", "amazing", "good", "great", "cool", "best", "love", "loved", "like", "liked",
    "stars", "star", "rating", "school", "class", "club", "giveaway", "arc", "netgalley", "imax",
    "genres", "listed", "no", "other", "misc", "general", "unknown",
    "favoritos", "favoritas", "favorito", "lidos", "leidos", "livros", "libros", "lido", "leido",
    "queue", "audible", "owned-books",
}
# tags genéricas demais para diferenciar itens
_STOP_EXACT = {"fiction", "adult", "adult-fiction", "general-fiction"}
_DECADE = re.compile(r"^\d0s$")
_NON_TAG = re.compile(r"[^a-z0-9]+")


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def normalize_tag(raw: str) -> list[str]:
    """Normaliza uma tag crua. Devolve zero, uma ou mais tags canônicas."""
    text = _ascii(raw.strip().lower()).replace("'", "")
    text = text.replace("r&b", "r-and-b").replace("&", "-and-")
    tag = _NON_TAG.sub("-", text).strip("-")
    if not tag:
        return []
    if tag in SYNONYMS:
        return list(SYNONYMS[tag])
    if _DECADE.match(tag):
        return [tag]
    parts = tag.split("-")
    if tag in _STOP_EXACT or any(part in _SHELF_PARTS for part in parts):
        return []
    if len(tag) < 3 or len(tag) > 28 or len(parts) > 3 or any(ch.isdigit() for ch in tag):
        return []
    return [tag]


def aggregate_tags(pairs: Iterable[tuple[str, float]], max_tags: int = 15) -> tuple[list[str], list[float]]:
    """Normaliza pares (tag crua, peso), soma pesos por tag canônica e devolve as mais fortes.

    Os pesos saem divididos pelo maior, em (0, 1].
    """
    totals: dict[str, float] = defaultdict(float)
    for raw, weight in pairs:
        for tag in normalize_tag(raw):
            totals[tag] += float(weight)
    if not totals:
        return [], []
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[:max_tags]
    top = ranked[0][1]
    return [t for t, _ in ranked], [round(w / top, 4) for _, w in ranked]
