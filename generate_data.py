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

    num_histories: int = 2
    episode_max_t: int = 20
    max_episodes: int = 1000

    gamma = 0.99
    hidden_dim = 32

    lr = 1e-4
    device = 'cpu'
    size: int = 9
    seed: int = 42

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def dump_trajectory(path: Path, trajectory):
    np.savez(path, trajectory)


def calculate_return(rewards: np.array, gamma: float):
    r = 0
    for reward in rewards[::-1]:
        r = reward + gamma * r
    return r


@pyrallis.wrap()
def collect(cfg: DataCollectionConfig):

    set_seed(cfg.seed)

    for i in range(cfg.num_histories):
        run = wandb.init(project=cfg.project, group=cfg.group, name=cfg.name, config=asdict(cfg), id=str(uuid.uuid4()))

        history_meta = dict()
        history_path = Path(os.path.join(cfg.data_path, f"history_{i}")).resolve()
        history_path.mkdir(parents=True, exist_ok=True)

        env = DarkRoom(cfg.size)

        global_ac = A3C(state_dim=env.state_dim, action_dim=env.action_dim, hidden_dim=cfg.hidden_dim, gamma=cfg.gamma,
                        device=cfg.device)
        global_ac.share_memory()

        queue = mp.Queue()
        episode_idx = mp.Value('i', 0)
        optim = SharedAdam(global_ac.parameters(), lr=cfg.lr)

        trajectory_counter = 0

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

        # Collect all trajectories from workers and dump them + log episode return
        while episode_idx.value < cfg.max_episodes or not queue.empty():
            if not queue.empty():
                trajectory = queue.get()
                trajectory_path = history_path / f"trajectory_{trajectory_counter}.npz"
                history_meta[trajectory_counter] = str(trajectory_path)

                dump_trajectory(trajectory_path, trajectory)

                trajectory_counter += 1

                r = calculate_return(trajectory['rewards'], cfg.gamma)
                wandb.log({"episode_return": r})

        [w.join() for w in workers]

        history_meta_path = history_path / f"meta.json"
        with open(history_meta_path, "w") as f:
            json.dump(history_meta, f)

        run.finish()


if __name__ == "__main__":
    collect()
