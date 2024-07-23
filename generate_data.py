import uuid
from dataclasses import dataclass, asdict
import numpy as np
import random

from envs import DarkRoom

import pyrallis
from pathlib import Path
import wandb
from tqdm import tqdm
import json

@dataclass
class DataCollectionConfig:
    project: str = "icl"
    group: str = "collect_data"
    env_name: str = "darkroom"
    data_path: str = "data"

    num_histories: int = 10
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


@pyrallis.wrap()
def generate(cfg: DataCollectionConfig):

    env = DarkRoom(size=cfg.size, random_start=cfg.random_start, terminate_on_goal=cfg.terminate_on_goal)
    set_seed(cfg.seed)
    env.reset(seed=cfg.seed)

    eps_diff = 1.0 / (0.9 * (cfg.num_timesteps // cfg.max_episode_steps))

    data_path = Path(cfg.data_path).resolve()
    data_path.mkdir(exist_ok=True, parents=True)

    for history_id in range(cfg.num_histories):

        run = wandb.init(project=cfg.project, name=cfg.name, group=cfg.group, config=asdict(cfg))

        history_meta = {}
        history_path = data_path / f"history_{history_id}"
        history_path.mkdir(exist_ok=True)
        history_meta_path = history_path / f"meta.json"

        episode_counter = 1

        q_learning = QLearning(cfg.state_dim, cfg.action_dim, lr=cfg.lr, gamma=cfg.gamma, epsilon=cfg.epsilon)
        env.generate_goal()
        state, _ = env.reset()

        buffer = RolloutBuffer()

        for timestep_counter in tqdm(range(1, cfg.num_timesteps + 1)):
            action = q_learning.act(state)
            next_state, reward, done, _, _ = env.step(action)

            buffer.add(state, action, reward, done)

            q_learning.update(state, action, reward, next_state)

            if timestep_counter % cfg.max_episode_steps == 0:

                q_learning.epsilon = max(0, q_learning.epsilon - eps_diff)

                trajectory = buffer.get_trajectory()
                trajectory_path = history_path / f"trajectory_{episode_counter}"
                np.savez(trajectory_path, trajectory)

                history_meta[episode_counter] = str(trajectory_path)
                wandb.log({"score": trajectory["rewards"].sum()})

                next_state, _ = env.reset()
                buffer.clear()
                episode_counter += 1

            state = next_state

        with open(str(history_meta_path), "w") as f:
            f.write(json.dumps(history_meta))
        wandb.finish()

if __name__ == "__main__":
    generate()