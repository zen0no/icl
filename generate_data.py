import uuid
from random import random

import pyrallis
from dataclasses import dataclass, asdict
import wandb
from wandb.wandb_torch import torch

from src.a3c import Worker, A3C, SharedAdam
from src.env import DarkRoom

import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import numpy as np
import os
import json


@dataclass
class DataCollectionConfig:
    project: str = "icl"
    group: str = "collect_data"
    name: str = "a3c_collect_data"

    data_path: str = "data"

    num_histories: int = 500
    train_steps: int = 1000

    episode_max_t: int = 20
    max_episodes: int = 50

    gamma = 0.99
    lambda_ = 0.95

    lr=3e-3
    device='cpu'
    size: int=9

    def __post__init__(self):
        self.name = f""


def wandb_init(cfg: DataCollectionConfig):
    wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg), id=str(uuid.uuid4()))


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def dump_trajectory(path: Path, trajectory):
    path.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def collect(cfg: DataCollectionConfig):
    for i in tqdm(range(cfg.num_histories)):
        wandb_init(cfg)

        history_meta = dict()
        history_path = Path(os.path.join(cfg.data_path,f"history_{i}")).resolve()

        env = DarkRoom(cfg.size)

        global_ac = A3C(env.state_dim, env.action_dim, cfg.gamma)
        global_ac.share_memory()

        queue = mp.Queue()
        episode_idx = mp.Value('i', 0)
        optim = SharedAdam(global_ac.parameters(), lr=cfg.lr)

        workers = [Worker(global_ac=global_ac,
                optimizer=optim,
                env=env.copy(),
                gamma=cfg.gamma,
                name=str(i),
                global_ep=episode_idx,
                data_queue=queue,
                t_max=cfg.episode_max_t,
                max_episode=cfg.max_episodes) for i in range(mp.cpu_count())]

        [w.start() for w in workers]
        while episode_idx.value < cfg.max_episodes:
            if not queue.empty():
                with episode_idx.lock:
                    trajectory_path = history_path / f"trajectory_{episode_idx.value}.npy"
                    trajectory = queue.get()

                    dump_trajectory(trajectory_path, trajectory)
                    history_meta[episode_idx.value] = trajectory_path
        [w.join() for w in workers]

        history_meta_path = history_path / f"meta.json"
        with open(history_meta_path, "w") as f:
            json.dump(history_meta, f)


if __name__ == "__main__":
    collect()

        


