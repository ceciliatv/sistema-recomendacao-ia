"""CLI `repertorio`: dados → treino → avaliação → uso."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from repertorio.config import get_paths
from repertorio.data.schema import DOMAINS, INTERACTION_DOMAINS

app = typer.Typer(help="Repertório: recomendação de livros, filmes, séries e música.", no_args_is_help=True)
data_app = typer.Typer(help="Baixar e preparar os datasets.", no_args_is_help=True)
eval_app = typer.Typer(help="Avaliação offline dos modelos e evals do agente.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
console = Console()


def _log(message: str) -> None:
    console.print(message, highlight=False)


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


# ------------------------------------------------------------------ dados e treino
@data_app.command("download")
def data_download() -> None:
    """Baixa MovieLens, Goodbooks-10k, Last.fm e TVMaze para data/raw/."""
    from repertorio.data.download import download_all

    download_all(get_paths().raw, _log)


@data_app.command("build")
def data_build() -> None:
    """Gera o catálogo único e as tabelas de interação em data/processed/."""
    from repertorio.data.build import build_all

    paths = get_paths()
    build_all(paths.raw, paths.processed, _log)


@app.command()
def train(domain: list[str] = typer.Option(list(INTERACTION_DOMAINS), help="Domínios a treinar.")) -> None:
    """Treina ItemKNN e Two-Tower com todos os dados e salva em artifacts/."""
    from repertorio.models.store import train_and_save

    for d in domain:
        train_and_save(get_paths(), d, _log)


# ------------------------------------------------------------------ avaliação
@eval_app.command("offline")
def eval_offline(
    protocol: str = typer.Option("loo", help="loo, new-users ou all"),
    domain: list[str] = typer.Option(list(INTERACTION_DOMAINS)),
) -> None:
    """Compara popularidade, ItemKNN e Two-Tower e escreve reports/offline_<protocolo>.md."""
    from repertorio.evals.offline import PROTOCOLS, run_offline

    protocols = list(PROTOCOLS) if protocol == "all" else [protocol]
    for p in protocols:
        if p not in PROTOCOLS:
            _fail(f"protocolo desconhecido: {p}")
        run_offline(get_paths(), p, domain, _log)


@eval_app.command("llm")
def eval_llm(
    provider: str = typer.Option("heuristic", help="anthropic, openai ou heuristic"),
    model: str | None = typer.Option(None, help="Sobrescreve o modelo do .env."),
    judge: str = typer.Option("none", help="none, auto (provedor oposto), anthropic ou openai"),
    only: list[str] = typer.Option([], help="Ids de caso ou categorias."),
    limit: int | None = typer.Option(None),
    yes: bool = typer.Option(False, "--yes", "-y", help="Não pede confirmação de custo."),
) -> None:
    """Roda os casos de evals/cases/ e escreve o relatório em reports/llm/."""
    from repertorio.agent.providers.base import ProviderError
    from repertorio.evals.llm.runner import run_llm_eval, select_cases

    paths = get_paths()
    cases = select_cases(paths, only or None, limit)
    judge_name = None if judge == "none" else judge
    paid = provider != "heuristic" or (judge_name is not None and judge_name != "heuristic")
    if paid and not yes:
        calls = f"{len(cases)} conversas" + (f" + {len(cases)} chamadas do juiz" if judge_name else "")
        typer.confirm(f"Isto faz {calls} em APIs pagas. Continuar?", abort=True)
    try:
        summary = run_llm_eval(paths, provider, model, judge_name, cases, _log)
    except ProviderError as exc:
        _fail(str(exc))
        return
    _log(f"\naprovação: {summary['pass_rate']} · relatório em {summary['folder']}")


# ------------------------------------------------------------------ uso
def _items_table(rows: list[dict], title: str) -> Table:
    table = Table(title=title, show_lines=False)
    for column in ("item_id", "título", "ano", "tags", "por quê"):
        table.add_column(column, overflow="fold")
    for r in rows:
        table.add_row(
            r["item_id"], r["title"], str(r.get("year") or ""), ", ".join(r.get("tags", [])[:4]), r.get("why", "")
        )
    return table


@app.command()
def search(
    query: str,
    domain: str | None = typer.Option(None, help=f"Um de: {', '.join(DOMAINS)}"),
    limit: int = 10,
) -> None:
    """Busca textual no catálogo."""
    from repertorio.runtime import load_engine

    engine = load_engine(get_paths())
    rows = [
        engine.catalog.summary(engine.catalog.item_id(r)) for r, _ in engine.content.search(query, domain, None, limit)
    ]
    console.print(_items_table(rows, f"busca: {query}"))


@app.command()
def recommend(
    like: list[str] = typer.Option([], help="Item curtido: id ou 'domínio:Título'. Repita a opção."),
    dislike: list[str] = typer.Option([]),
    domain: str = typer.Option("book", help=f"Um de: {', '.join(DOMAINS)}"),
    k: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    exclude: list[str] = typer.Option([], help="Tags a evitar."),
) -> None:
    """Recomendações do motor, sem LLM."""
    from repertorio.profile import TasteProfile
    from repertorio.retrieval.content import Filters
    from repertorio.runtime import load_engine

    engine = load_engine(get_paths())
    try:
        profile = TasteProfile.from_references(engine.catalog, like, dislike)
    except KeyError as exc:
        _fail(str(exc))
        return
    liked = ", ".join(engine.catalog.get(i)["title"] for i in profile.liked) or "(perfil vazio)"
    recs = engine.recommend(profile, domain, k, Filters(year_min, year_max, exclude))
    rows = [engine.catalog.summary(r.item_id) | {"why": "; ".join(r.reasons)} for r in recs]
    console.print(_items_table(rows, f"{domain} para quem curtiu: {liked}"))


@app.command()
def chat(
    provider: str = typer.Option("anthropic", help="anthropic, openai ou heuristic"),
    model: str | None = None,
    profile_path: Path = typer.Option(Path("profile.json"), help="Onde o perfil é lido e salvo."),
) -> None:
    """Conversa com o agente. O perfil é salvo entre as sessões."""
    from repertorio.agent.agent import RecommenderAgent
    from repertorio.agent.providers import get_provider
    from repertorio.agent.providers.base import ProviderError
    from repertorio.profile import TasteProfile
    from repertorio.runtime import load_engine

    engine = load_engine(get_paths())
    profile = TasteProfile.load(profile_path)
    try:
        agent = RecommenderAgent(get_provider(provider, model), engine, profile)
    except ProviderError as exc:
        _fail(str(exc))
        return
    _log(f"[bold]Repertório[/bold] ({provider}). Diga o que você curte e o que quer. Ctrl+C para sair.")
    while True:
        try:
            message = console.input("[bold cyan]você>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        try:
            result = agent.ask(message)
        except ProviderError as exc:
            _log(f"[red]{exc}[/red]")
            continue
        if result.message:
            _log(f"[bold]repertório>[/bold] {result.message}")
        for i, rec in enumerate(result.recommendations, 1):
            _log(f"  {i}. [bold]{rec['title']}[/bold] ({rec['domain']}) — {rec['reason']}")
        if result.error:
            _log(f"[yellow]{result.error}[/yellow]")
        agent.profile.save(profile_path)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Sobe a API HTTP (FastAPI)."""
    import uvicorn

    uvicorn.run("repertorio.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
