import pyrallis
from dataclasses import dataclass
import wandb

from src.a3c import Worker, A3C, SharedAdam
from src.env import DarkRoom

import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import numpy as np


@dataclass
class DataCollectionConfig:
    team: str = "zen0no"
    project: str = "icl-id"
    name: str = "a3c_collect_data"

    num_histories: int = 500
    train_steps: int = 1000

    episode_max_t: int = 20
    max_episods: int = 50

    gamma = 0.99
    lambda_ = 0.95


    def __post__init__(self):
        self.name = f""


def wandb_init(cfg: DataCollectionConfig):
    pass


def define_wandb_metric():
    wandb.define_metric("")


def dump_trajectory(path: Path, trajectory):

    

@pyrallis.wrap
def collect(cfg: DataCollectionConfig):

    wandb_init(cfg)
    define_wandb_metric()

    for i in tqdm(range(num_l))


    


