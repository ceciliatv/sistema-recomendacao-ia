"""Two-Tower com pooling de histórico, em PyTorch.

Torre do usuário: média dos embeddings dos itens do histórico + MLP residual.
Torre do item: tabela de embeddings + viés (que aprende a popularidade).

Como a torre do usuário só depende do histórico, o modelo é indutivo: uma pessoa
nova, que diz três livros de que gostou, recebe um vetor na hora, sem re-treino.

Treino: para cada par (usuário, item positivo), o histórico de entrada é uma amostra
de tamanho aleatório dos outros positivos do usuário, e a perda é a entropia cruzada
do softmax sobre o catálogo inteiro, com os demais itens do histórico mascarados
(eles também são positivos e não devem ser empurrados para baixo).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from repertorio.models.base import Recommender


@dataclass
class TwoTowerConfig:
    dim: int = 64
    max_history: int = 50
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-5
    dropout: float = 0.1
    seed: int = 42
    device: str = "cpu"


class TwoTowerModule(nn.Module):
    def __init__(self, n_items: int, dim: int, dropout: float):
        super().__init__()
        self.n_items = n_items
        self.item_in = nn.Embedding(n_items + 1, dim, padding_idx=n_items)  # índice n_items = padding
        self.user_mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.item_out = nn.Embedding(n_items, dim)
        self.item_bias = nn.Parameter(torch.zeros(n_items))
        nn.init.normal_(self.item_in.weight, std=0.05)
        nn.init.normal_(self.item_out.weight, std=0.05)
        with torch.no_grad():
            self.item_in.weight[n_items].zero_()

    def user_vectors(self, history: torch.Tensor) -> torch.Tensor:
        mask = (history != self.n_items).unsqueeze(-1).float()
        pooled = (self.item_in(history) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return pooled + self.user_mlp(pooled)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.user_vectors(history) @ self.item_out.weight.T + self.item_bias


class TwoTowerRecommender(Recommender):
    name = "two_tower"

    def __init__(self, config: TwoTowerConfig | None = None):
        super().__init__()
        self.config = config or TwoTowerConfig()
        self.module: TwoTowerModule | None = None
        self.losses: list[float] = []

    # ------------------------------------------------------------------ treino
    def fit(self, histories: list[np.ndarray], n_items: int, log=None) -> TwoTowerRecommender:
        cfg = self.config
        self.n_items = n_items
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        device = torch.device(cfg.device)
        self.module = TwoTowerModule(n_items, cfg.dim, cfg.dropout).to(device)
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        usable = [h for h in histories if len(h) >= 2]
        pairs = np.array([(u, pos) for u, h in enumerate(usable) for pos in range(len(h))], dtype=np.int64)
        steps_per_epoch = int(np.ceil(len(pairs) / cfg.batch_size))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=max(1, steps_per_epoch * cfg.epochs)
        )

        self.module.train()
        for epoch in range(cfg.epochs):
            order = rng.permutation(len(pairs))
            total = 0.0
            for start in range(0, len(order), cfg.batch_size):
                batch = pairs[order[start : start + cfg.batch_size]]
                history, target, others = self._make_batch(usable, batch, rng)
                logits = self.module(history.to(device))
                # outros positivos do histórico completo não contam como negativos
                logits = logits.masked_fill(others.to(device), float("-inf"))
                loss = nn.functional.cross_entropy(logits, target.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item() * len(batch)
            self.losses.append(total / len(pairs))
            if log:
                log(f"  época {epoch + 1}/{cfg.epochs}  perda={self.losses[-1]:.4f}")
        self.module.eval()
        return self

    def _make_batch(self, histories, batch, rng):
        n, pad, cap = self.n_items, self.n_items, self.config.max_history
        history = np.full((len(batch), cap), pad, dtype=np.int64)
        target = np.empty(len(batch), dtype=np.int64)
        others = np.zeros((len(batch), n), dtype=bool)
        for row, (user, pos) in enumerate(batch):
            items = histories[user]
            target[row] = items[pos]
            rest = np.delete(items, pos)
            others[row, rest] = True
            # tamanho aleatório: ensina o modelo a lidar com perfis de 1 ou 2 itens
            size = int(rng.integers(1, min(cap, len(rest)) + 1))
            history[row, :size] = rng.choice(rest, size=size, replace=False)
        return torch.from_numpy(history), torch.from_numpy(target), torch.from_numpy(others)

    # --------------------------------------------------------------- inferência
    def _pad(self, histories: list[np.ndarray]) -> torch.Tensor:
        cap = self.config.max_history
        out = np.full((len(histories), cap), self.n_items, dtype=np.int64)
        for row, items in enumerate(histories):
            items = np.asarray(items, dtype=np.int64)[-cap:]
            out[row, : len(items)] = items
        return torch.from_numpy(out)

    @torch.no_grad()
    def score(self, histories: list[np.ndarray]) -> np.ndarray:
        assert self.module is not None, "modelo não treinado"
        device = next(self.module.parameters()).device
        return self.module(self._pad(histories).to(device)).float().cpu().numpy()

    @torch.no_grad()
    def item_similarity(self, item: int) -> np.ndarray:
        assert self.module is not None, "modelo não treinado"
        emb = nn.functional.normalize(self.module.item_out.weight, dim=1)
        return (emb @ emb[item]).cpu().numpy()

    # ---------------------------------------------------------------- arquivos
    def save(self, folder: Path) -> None:
        assert self.module is not None, "modelo não treinado"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), folder / "two_tower.pt")
        meta = {"config": asdict(self.config), "n_items": self.n_items, "losses": self.losses}
        (folder / "two_tower.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, folder: Path) -> TwoTowerRecommender:
        meta = json.loads((folder / "two_tower.json").read_text())
        config = TwoTowerConfig(**{**meta["config"], "device": "cpu"})
        model = cls(config)
        model.n_items = meta["n_items"]
        model.losses = meta.get("losses", [])
        model.module = TwoTowerModule(model.n_items, config.dim, config.dropout)
        state = torch.load(folder / "two_tower.pt", map_location="cpu", weights_only=True)
        model.module.load_state_dict(state)
        model.module.eval()
        return model
