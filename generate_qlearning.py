import json
import os
import random
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

import envs
import gym
import gymnasium as gym
import numpy as np
import pyrallis
from tqdm import tqdm

import wandb
from src.algorithm import QLearning, RolloutBuffer


@dataclass
class DataCollectionConfig:
    # wandb
    project: str = "icl"
    group: str = "collect_data"
    env_name: str = "DarkRoom"
    algorithm: str = "q_learning"
    data_path: str = "data"

    # training
    num_train_envs: int = 5
    num_train_episodes: int = 20_000
    gamma = 0.99
    epsilon = 1.
    lr = 1e-4
    seed: int = 42

    # env properties
    size: int = 9
    terminate_on_goal: bool = False
    random_start: bool = True
    max_episode_steps: int = 20
    action_dim: int = 5

    def __post_init__(self):
        self.state_dim = self.size ** 2
        self.name = f"{self.env_name}_{self.group}_{uuid.uuid4()}"
        self.data_path = os.path.join(self.data_path, f"{self.env_name}_{self.algorithm}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def dump_meta(path: Path, meta: dict):
    with open(str(path), 'w') as f:
        json.dump(meta, f)


def dump_trajectory(path, traj):
    np.savez(path,
             states=np.uint8(traj["states"]),
             actions=np.uint8(traj["states"]),
             rewards=np.uint8(traj["rewards"]),
             dones=np.uint8(traj["terminated"] | traj["truncated"]))


@pyrallis.wrap()
def generate(cfg: DataCollectionConfig):

    env = gym.make(cfg.env_name, max_episode_steps=cfg.max_episode_steps, size=cfg.size, random_start=cfg.random_start, terminate_on_goal=cfg.terminate_on_goal)
    set_seed(cfg.seed)
    env.reset(seed=cfg.seed)

    meta = {
        "env_name": cfg.env_name,
        "state_dim": env.unwrapped.state_dim,
        "action_dim": env.unwrapped.action_dim,
        "max_episode_steps": cfg.max_episode_steps,
        "size": cfg.size,
        "random_start": cfg.random_start,
        "terminate_on_goal": cfg.terminate_on_goal,
        "num_train_envs": cfg.num_train_envs,
        "num_train_episodes": cfg.num_train_episodes
    }

    eps_diff = 1.0 / (0.9 * cfg.num_train_episodes)

    data_path = Path(cfg.data_path).resolve()
    data_path.mkdir(exist_ok=True, parents=True)

    meta_path = data_path / "meta.json"
    dump_meta(meta_path, meta)

    for history_id in range(cfg.num_train_envs):

        run = wandb.init(project=cfg.project, name=cfg.name, group=cfg.group, config=asdict(cfg))

        history_path = data_path / f"history_{history_id}"
        history_path.mkdir(exist_ok=True)

        episode_counter = 0

        q_learning = QLearning(cfg.state_dim, cfg.action_dim, lr=cfg.lr, gamma=cfg.gamma, epsilon=cfg.epsilon)
        env.unwrapped.generate_goal()
        state, _ = env.reset()

        buffer = RolloutBuffer()

        while True:
            action = q_learning.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            buffer.add(state, action, reward, terminated, truncated)

            q_learning.update(state, action, reward, next_state)

            if terminated or truncated:

                q_learning.epsilon = max(0, q_learning.epsilon - eps_diff)

                trajectory = buffer.get_trajectory()
                trajectory_path = history_path / f"{episode_counter}.npz"
                dump_trajectory(trajectory_path, trajectory)

                wandb.log({"score": trajectory["rewards"].sum()})

                next_state, _ = env.reset()
                buffer.clear()
                episode_counter += 1
                if episode_counter >= cfg.num_train_episodes:
                    break

            state = next_state
        run.finish()


if __name__ == "__main__":
    generate()
