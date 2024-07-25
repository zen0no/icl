import uuid
from dataclasses import dataclass, asdict

import gym
import numpy as np
import random

from envs import DarkRoom
import gymnasium as gym
from src.algorithm import QLearning, RolloutBuffer
import pyrallis
from pathlib import Path
import wandb
from tqdm import tqdm

@dataclass
class DataCollectionConfig:
    project: str = "icl"
    group: str = "collect_data"
    env_name: str = "darkroom"
    data_path: str = "data"

    num_histories: int = 1
    num_timesteps: int = int(1e6)
    gamma = 0.99
    epsilon = 1.

    lr = 1e-4
    size: int = 9
    terminate_on_goal: bool = False
    random_start: bool = True
    max_episode_steps: int = 20
    action_dim: int = 5
    seed: int = 42

    def __post_init__(self):
        self.state_dim = self.size ** 2
        self.name = f"{self.env_name}_{self.group}_{uuid.uuid4()}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def dump_trajectory(path, traj):
    np.savez(path,
             states=np.uint8(traj["states"]),
             actions=np.uint8(traj["states"]),
             rewards=np.uint8(traj["rewards"]),
             dones=np.uint8(traj["terminated"] | traj["truncated"]))

@pyrallis.wrap()
def generate(cfg: DataCollectionConfig):

    env = gym.make("DarkRoom", max_episode_steps=20, size=cfg.size, random_start=cfg.random_start, terminate_on_goal=cfg.terminate_on_goal)
    set_seed(cfg.seed)
    env.reset(seed=cfg.seed)

    eps_diff = 1.0 / (0.9 * (cfg.num_timesteps // cfg.max_episode_steps))

    data_path = Path(cfg.data_path).resolve()
    data_path.mkdir(exist_ok=True, parents=True)

    for history_id in range(cfg.num_histories):

        run = wandb.init(project=cfg.project, name=cfg.name, group=cfg.group, config=asdict(cfg))

        history_path = data_path / f"history_{history_id}"
        history_path.mkdir(exist_ok=True)

        episode_counter = 0

        q_learning = QLearning(cfg.state_dim, cfg.action_dim, lr=cfg.lr, gamma=cfg.gamma, epsilon=cfg.epsilon)
        env.generate_goal()
        state, _ = env.reset()

        buffer = RolloutBuffer()

        for timestep_counter in tqdm(range(1, cfg.num_timesteps + 1)):
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

            state = next_state
        wandb.finish()


if __name__ == "__main__":
    generate()
