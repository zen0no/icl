import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from src.transformer import Transformer
from src.data import SequentialDataset

from dataclasses import dataclass, asdict

import pyrallis
from tqdm import trange

from typing import Tuple
from pathlib import Path

from src.utils import cosine_annealing_with_warmup


@dataclass
class TrainConfig:
    # wandb
    project: str = "icl"
    group: str = "algorithm_distillation"
    seed: int = 42
    device: str = "cuda"

    # data
    data_path: str = "data"
    max_episode_steps: int = 20
    num_train_envs: int = 20_000

    # model
    num_attn_head: int = 4
    num_layers: int = 4
    attn_dropout_rate: float = 0.3
    res_dropout_rate: float = 0.
    feed_forward_dim = 2048

    # train
    train_context_multiplier: float = 4.
    batch_size: int = 32
    masking_prob: float = 0.3
    lr: float = 3e-4
    num_train_steps: int = 100_000
    warmap_steps: int = 20_000

    # evaluation
    eval_env_name: str = "DarkRoom"
    eval_context_multipliers: Tuple[float] = (0.5, 1., 2., 4.)
    eval_log_every: int = 10
    eval_num_episodes: int = 1000
    num_eval_envs: int = 10

    def __post_init__(self):
        self.train_seed: int = self.seed
        self.valid_seed: int = 100 + self.seed
        self.train_seq_len: int = int(self.max_episode_steps * self.train_context_multiplier)

        self.train_run_name = f"train_"


@torch.no_grad()
def eval_in_context(config: TrainConfig):
    pass


@pyrallis.wrap()
def train(config: TrainConfig):

    dataset = SequentialDataset(data_path=Path(config.data_path), seq_len=config.train_seq_len)
    model = Transformer(

    ).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = cosine_annealing_with_warmup(optimizer,
                                             warmup_steps=config.warmap_steps,
                                             total_steps=config.num_train_steps)

    dataloader = DataLoader()


if __name__ == "__main__":
    train()