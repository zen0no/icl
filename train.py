import gc

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.transformer import Transformer
from src.data import SequentialDataset

from dataclasses import dataclass, asdict

import pyrallis
from tqdm import trange

from typing import Tuple
from pathlib import Path

from src.utils.scheduler import cosine_annealing_with_warmup
from src.utils.envs import make_eval_envs


@dataclass
class TrainConfig:
    # wandb
    project: str = "icl"
    group: str = "algorithm_distillation"
    seed: int = 42
    device: str = "cpu"

    # data
    data_path: str = r"data\DarkRoom_q_learning"
    max_episode_steps: int = 20
    num_train_envs: int = 20_000

    # model
    num_attn_head: int = 4
    num_layers: int = 4
    attn_dropout_rate: float = 0.3
    res_dropout_rate: float = 0.
    embed_dropout_rate: float = 0.
    feed_forward_dim = 2048
    hidden_dim: int = 64

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


def iter_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def eval_in_context(config: TrainConfig, env_props: dict):
    envs = make_eval_envs(config.num_eval_envs, config.eval_env_name, **env_props)

    for multiplier in config.eval_context_multipliers:
        seq_len = config.max_episode_steps * multiplier

        states = torch.zeros(config.state)


@pyrallis.wrap()
def train(config: TrainConfig):

    dataset = SequentialDataset(
        data_path=Path(config.data_path),
        seq_len=config.train_seq_len,
        masking_prob=config.masking_prob,
        device=config.device
    )
    model = Transformer(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_dim=config.hidden_dim,
        feedforward_dim=config.feed_forward_dim,
        seq_len=config.train_seq_len,
        num_blocks=config.num_layers,
        num_attention_heads=config.num_attn_head,
        attn_dropout=config.attn_dropout_rate,
        res_dropout=config.res_dropout_rate,
        embed_dropout=config.embed_dropout_rate
    ).to(config.device)


    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = cosine_annealing_with_warmup(optimizer,
                                             warmup_steps=config.warmap_steps,
                                             total_steps=config.num_train_steps)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
    )

    dataloader = iter_dataloader(dataloader)

    for train_step in trange(0, config.num_train_steps, desc='Train'):

        states, actions, rewards, masks = next(dataloader)

        states = states.to(config.device)
        actions = actions.to(config.device)
        rewards = rewards.to(config.device)
        masks = masks.to(config.device)

        pred = model(
            states=states,
            actions=actions,
            rewards=rewards,
            padding_mask=masks
        )

        loss = F.cross_entropy(pred, actions)

        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()



if __name__ == "__main__":
    train()