"""Agente por regras, sem LLM. É a linha de base dos evals e roda sem chave de API.

Segue o mesmo protocolo dos LLMs (chama ferramentas e entrega pelo
`submit_recommendations`), então os mesmos graders valem para os dois. A
interpretação do pedido é feita com expressões regulares em português e inglês.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from repertorio.agent.providers.base import (
    AssistantTurn,
    Conversation,
    Provider,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from repertorio.data.tags import normalize_tag

_DOMAIN_WORDS = {
    "book": r"livros?|books?|ler|leitura|romances? para ler|autora?s?",
    "movie": r"filmes?|movies?|films?|cinema",
    "series": r"s[ée]ries?|seriados?|shows?|temporadas?|tv",
    "music": r"m[úu]sicas?|music|bandas?|bands?|artistas?|artists?|[áa]lbu[mn]s?|cantor(?:a|es)?|ouvir|playlist",
}
_NUMBER_WORDS = {
    "um": 1,
    "uma": 1,
    "one": 1,
    "a": 1,
    "dois": 2,
    "duas": 2,
    "two": 2,
    "três": 3,
    "tres": 3,
    "three": 3,
    "quatro": 4,
    "four": 4,
    "cinco": 5,
    "five": 5,
    "seis": 6,
    "six": 6,
    "sete": 7,
    "seven": 7,
    "oito": 8,
    "eight": 8,
    "nove": 9,
    "nine": 9,
    "dez": 10,
    "ten": 10,
}
_ANCHOR = re.compile(
    r"(?:parecid[oa]s? com|semelhantes? a|no estilo de|na vibe de|similar to|something like|anything like"
    r"|(?:books?|movies?|films?|shows?|series|music|artists?|bands?) like)\s+[\"“']?([^\"”',.?!;]+)",
    re.IGNORECASE,
)
_EXCLUDE = re.compile(r"(?:sem|nada de|não quero|nao quero|without|não gosto de|odeio)\s+([\wÀ-ú-]+)", re.IGNORECASE)


# gêneros e humores que o baseline reconhece no texto e passa como `require_tags`
GENRE_TAGS = {
    "science-fiction",
    "fantasy",
    "horror",
    "romance",
    "comedy",
    "thriller",
    "mystery",
    "crime",
    "drama",
    "dystopia",
    "dark",
    "animation",
    "documentary",
    "biography",
    "historical",
    "war",
    "western",
    "adventure",
    "action",
    "kids",
    "young-adult",
    "supernatural",
    "space",
    "magic",
    "classic",
    "melancholic",
    "epic",
    "rock",
    "pop",
    "jazz",
    "hip-hop",
    "electronic",
    "metal",
    "classical",
    "indie",
    "brazilian",
    "mpb",
    "chill",
    "folk",
    "punk",
    "soul",
    "rnb",
    "blues",
    "country",
    "reggae",
    "ambient",
}
_PHRASES = {"ficção científica": "sci-fi", "ficcao cientifica": "sci-fi", "science fiction": "sci-fi"}


@dataclass
class Plan:
    domains: list[str]
    count: int
    year_min: int | None = None
    year_max: int | None = None
    exclude_tags: list[str] = field(default_factory=list)
    require_tags: list[str] = field(default_factory=list)
    anchor: str | None = None


def parse_request(text: str) -> Plan:
    lower = text.lower()
    positions = []
    for domain, pattern in _DOMAIN_WORDS.items():
        match = re.search(rf"\b(?:{pattern})\b", lower)
        if match:
            positions.append((match.start(), domain))
    domains = [d for _, d in sorted(positions)]

    count = None
    for match in re.finditer(r"\b(\d{1,2})\b(?!s)", lower):
        if int(match.group(1)) <= 20 and not lower[: match.start()].endswith("anos "):
            count = int(match.group(1))
            break
    if count is None:
        words = [w for w in re.findall(r"[a-zà-ú]+", lower) if w in _NUMBER_WORDS and w != "a"]
        count = _NUMBER_WORDS[words[0]] if words else None
    count = count or (2 if len(domains) > 1 else 5)

    plan = Plan(domains=domains, count=min(count, 10))
    if m := re.search(r"(?:depois de|após|a partir de|desde|after|since)\s+(\d{4})", lower):
        plan.year_min = int(m.group(1))
    if m := re.search(r"(?:antes de|before|até)\s+(\d{4})", lower):
        plan.year_max = int(m.group(1))
    if m := re.search(r"anos\s+(\d0)\b|\b(?:19)?(\d0)s\b", lower):
        decade = int(m.group(1) or m.group(2))
        start = 1900 + decade if decade >= 20 else 2000 + decade
        plan.year_min, plan.year_max = start, start + 9
    if re.search(r"recentes?|lançamentos?|recent|new releases", lower):
        plan.year_min = plan.year_min or 2015
    for m in _EXCLUDE.finditer(lower):
        plan.exclude_tags += normalize_tag(m.group(1))
    if m := _ANCHOR.search(text):
        plan.anchor = m.group(1).strip()
    for phrase, replacement in _PHRASES.items():
        lower = lower.replace(phrase, replacement)
    anchor = (plan.anchor or "").lower()
    for word in re.findall(r"[\wÀ-ú-]+", lower):
        if word in anchor:
            continue
        for tag in normalize_tag(word):
            if tag in GENRE_TAGS and tag not in plan.exclude_tags and tag not in plan.require_tags:
                plan.require_tags.append(tag)
    return plan


class HeuristicConversation(Conversation):
    def __init__(self) -> None:
        self.plan: Plan | None = None
        self.stage = 0
        self.results: list[dict] = []
        self.calls = 0

    def _call(self, name: str, **arguments) -> ToolCall:
        self.calls += 1
        return ToolCall(id=f"h{self.calls}", name=name, arguments={k: v for k, v in arguments.items() if v})

    def add_user(self, text: str) -> None:
        self.plan = parse_request(text)
        self.stage = 0
        self.results = []

    def add_tool_results(self, results: list[ToolResult]) -> None:
        self.results = [json.loads(r.content) if not r.is_error else {} for r in results]

    def step(self) -> AssistantTurn:
        plan = self.plan
        if plan is None:
            return AssistantTurn(text="")
        filters = {"year_min": plan.year_min, "year_max": plan.year_max, "exclude_tags": plan.exclude_tags}
        if self.stage == 0:
            if not plan.domains:
                self.stage = 3
                return AssistantTurn(text="Você quer recomendações de livros, filmes, séries ou música?")
            if plan.anchor:
                self.stage = 1
                return AssistantTurn(text="", tool_calls=[self._call("search_catalog", query=plan.anchor, limit=1)])
            self.stage = 2
            calls = [
                self._call("recommend_for_user", domain=d, k=plan.count, require_tags=plan.require_tags, **filters)
                for d in plan.domains
            ]
            return AssistantTurn(text="", tool_calls=calls)
        if self.stage == 1:
            found = (self.results[0].get("items") or [None])[0] if self.results else None
            self.stage = 2
            if found:
                calls = [
                    self._call("find_similar", item_id=found["item_id"], target_domain=d, k=plan.count, **filters)
                    for d in plan.domains
                ]
            else:
                calls = [self._call("recommend_for_user", domain=d, k=plan.count, **filters) for d in plan.domains]
            return AssistantTurn(text="", tool_calls=calls)
        if self.stage == 2:
            self.stage = 3
            picks = []
            for result in self.results:
                for item in (result.get("items") or [])[: plan.count]:
                    reason = (item.get("reasons") or ["resultado da busca"])[0]
                    picks.append({"item_id": item["item_id"], "reason": reason})
            message = "Aqui estão minhas sugestões." if picks else "Não encontrei itens com essas restrições."
            submit = self._call("submit_recommendations", recommendations=picks, message=message)
            submit.arguments.setdefault("recommendations", [])
            return AssistantTurn(text="", tool_calls=[submit])
        return AssistantTurn(text="")


class HeuristicProvider(Provider):
    name = "heuristic"
    model = "rules-v1"

    def start(self, system: str, tools: list[ToolSpec]) -> Conversation:
        return HeuristicConversation()
