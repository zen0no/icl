import gc
import os.path
import random
import uuid
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Tuple
from itertools import count

import uuid

import envs
import json
import os

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from gymnasium.vector import SyncVectorEnv
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from src.data import SequentialDataset
from src.transformer import Transformer
from src.utils.scheduler import cosine_annealing_with_warmup


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
    num_layers: int = 8
    attn_dropout_rate: float = 0.3
    res_dropout_rate: float = 0.
    embed_dropout_rate: float = 0.
    feed_forward_dim = 2048
    hidden_dim: int = 128

    # train
    train_context_multiplier: float = 4.
    batch_size: int = 32
    lr: float = 3e-4
    num_train_steps: int = 100_000
    warmap_steps: int = 20_000
    checkpoint_path: str = "checkpoint"

    # evaluation
    eval_env_name: str = "DarkRoom"
    eval_context_multipliers: Tuple[float] = (0.5, 1., 2., 4.)
    eval_log_every: int = 10
    eval_num_episodes: int = 1000
    eval_num_envs: int = 10

    def __post_init__(self):
        self.uuid: str = str(uuid.uuid4())
        self.train_seed: int = self.seed
        self.eval_seed: int = 100 + self.seed
        self.train_seq_len: int = int(self.max_episode_steps * self.train_context_multiplier)

        self.train_run_name = f"train_ad_{self.uuid}"
        self.eval_run_name = f"evaluate_ad_{self.eval_env_name}_{self.uuid}"
        self.checkpoint_path = os.path.join(self.checkpoint_path, self.uuid)


def set_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def iter_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@torch.no_grad()
def eval_in_context(model: Transformer, config: TrainConfig, env_props):
    for multiplier in config.eval_context_multipliers:
        set_seed(config.eval_seed)
        rng = np.random.default_rng(seed=config.eval_seed)

        num_envs = config.eval_num_envs
        eval_envs = SyncVectorEnv([
            partial(gym.make, id=config.eval_env_name, rng=rng, **env_props) for _ in range(num_envs)
        ])

        runs = [wandb.init(
            project=config.project,
            group=config.group,
            name=f"{config.eval_run_name}_{multiplier}_episode"
        ) for _ in range(num_envs)]

        seq_len = int(config.max_episode_steps * multiplier)

        states = torch.zeros(num_envs, seq_len, dtype=torch.long, device=config.device)
        actions = torch.zeros(num_envs, seq_len, dtype=torch.long, device=config.device)
        rewards = torch.zeros(num_envs, seq_len, 1, dtype=torch.float32, device=config.device)

        init_states, _ = eval_envs.reset()

        states[:, -1] = torch.from_numpy(init_states).to(config.device)

        num_dones = np.zeros(num_envs, dtype=np.int32)

        current_scores = np.zeros(num_envs)
        current_lengths = np.zeros(num_envs)

        for i in tqdm(count(start=1), desc="Evaluation"):
            sliced_states = states[:, -i:]
            sliced_actions = actions[:, -i:]
            sliced_rewards = rewards[:, -i:]
            timesteps = torch.arange(min(i, seq_len)).expand(num_envs, -1)

            pred = model(
                states=sliced_states,
                actions=sliced_actions,
                rewards=sliced_rewards,
                timesteps=timesteps
            )

            pred = pred[:, -1, :]

            dist = torch.distributions.Categorical(pred)
            action = dist.sample().squeeze(-1)

            state, reward, term, trunc, _ = eval_envs.step(action.cpu().numpy())

            for i in np.where(term | trunc)[0]:
                if num_dones[i] < config.eval_num_episodes:
                    runs[i].log({"score": current_scores[i]})
                    runs[i].log({"lenghts": current_lengths[i]})
                    current_scores[i] = 0
                    current_lengths[i] = 0

            states = states.roll(-1, dims=0)
            actions = actions.roll(-1, dims=0)
            rewards = rewards.roll(-1, dims=0)

            states[-1] = torch.from_numpy(state).type(torch.long).to(config.device)
            actions[-2] = action
            rewards[-2] = torch.from_numpy(reward).type(torch.long).to(config.device).unsqueeze(-1)

            if np.min(num_dones) >= config.eval_num_episodes:
                break

        [run.finish() for run in runs]


@pyrallis.wrap()
def train(config: TrainConfig):

    dataset = SequentialDataset(
        data_path=Path(config.data_path),
        seq_len=config.train_seq_len,
        device=config.device
    )

    set_seed(config.train_seed)

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

    print(f"Running model with {sum(p.numel() for p in model.parameters())} parameters")


    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = cosine_annealing_with_warmup(optimizer,
                                             warmup_steps=config.warmap_steps,
                                             total_steps=config.num_train_steps)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        pin_memory=True
    )

    dataloader = iter_dataloader(dataloader)

    run = wandb.init(project=config.project,
                     group=config.group,
                     name=config.train_run_name)

    for train_step in trange(0, config.num_train_steps, desc='Train'):

        states, actions, rewards, timesteps, masks = next(dataloader)

        states = states.to(config.device)
        actions = actions.to(config.device)
        rewards = rewards.to(config.device)
        timesteps = timesteps.to(config.device)
        masks = masks.to(config.device)

        pred = model(
            states=states,
            actions=actions,
            rewards=rewards,
            padding_mask=masks,
            timesteps=timesteps
        )
        pred = pred[~masks]
        action_idx = actions[~masks]

        loss = F.cross_entropy(pred, torch.eye(pred.shape[-1])[action_idx])
        run.log({"loss": loss.item()})

        loss.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        gc.collect()

    run.finish()

    # save train
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), checkpoint_path/"weights.pt")
    with open(str(checkpoint_path / "train_config.json"), 'r') as f:
        json.dump(asdict(config), f, indent=2)

    eval_in_context(model, config, dataset.meta)




if __name__ == "__main__":
    train()