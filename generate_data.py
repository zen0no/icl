import uuid
import random
import pyrallis
from dataclasses import dataclass, asdict
import wandb
import torch
from src.a3c import Worker, A3C, SharedAdam
from src.env import DarkRoom

import torch.multiprocessing as mp

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
    hidden_dim = 32

    lr = 3e-3
    device = 'cpu'
    size: int = 9
    seed: int = 42

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def dump_trajectory(path: Path, trajectory):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path, trajectory)


@pyrallis.wrap()
def collect(cfg: DataCollectionConfig):

    set_seed(cfg.seed)

    mp.Manager

    for i in range(cfg.num_histories):
        run = wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg), id=str(uuid.uuid4()))

        history_meta = dict()
        history_path = Path(os.path.join(cfg.data_path, f"history_{i}")).resolve()

        env = DarkRoom(cfg.size)

        global_ac = A3C(state_dim=env.state_dim, action_dim=env.action_dim, hidden_dim=cfg.hidden_dim, gamma=cfg.gamma,
                        device=cfg.device)
        global_ac.share_memory()

        queue = mp.Queue()
        episode_idx = mp.Value('i', 0)
        optim = SharedAdam(global_ac.parameters(), lr=cfg.lr)

        workers = [Worker(global_ac=global_ac,
                          optimizer=optim,
                          env=env.copy(),
                          gamma=cfg.gamma,
                          name=i,
                          global_ep=episode_idx,
                          data_queue=queue,
                          t_max=cfg.episode_max_t,
                          max_episode=cfg.max_episodes,
                          hidden_dim=cfg.hidden_dim,
                          device=cfg.device) for i in range(2)]

        [w.start() for w in workers]
        while episode_idx.value < cfg.max_episodes:
            if not queue.empty():
                with episode_idx.get_lock():
                    trajectory_path = history_path / f"trajectory_{episode_idx.value}.npy"
                    trajectory = queue.get()

                    dump_trajectory(trajectory_path, trajectory)
                    history_meta[episode_idx.value] = trajectory_path
        [w.join() for w in workers]

        history_meta_path = history_path / f"meta.json"
        with open(history_meta_path, "w") as f:
            json.dump(history_meta, f)

        run.finish()


if __name__ == "__main__":
    collect()
