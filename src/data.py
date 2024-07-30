import json

import numpy as np
import torch

import torch.multiprocessing as mp
from torch.utils.data import IterableDataset

from pathlib import Path


def load_single_history(history_path):
    trajectories = {}
    for p in history_path.glob("*.npz"):
        trajectory_dict = np.load(p, "r")
        traj = np.stack([trajectory_dict["states"],
                         trajectory_dict["actions"],
                         trajectory_dict["rewards"],
                         trajectory_dict["dones"]],
                        dtype=np.uint8)
        traj_id = int(p.stem)
        trajectories[traj_id] = traj

    sorted_items = sorted(trajectories.items(), key=lambda x: x[0])
    history = np.concatenate([x[1] for x in sorted_items], axis=-1)

    return history


def load_histories(data_path: Path):
    with mp.Pool(mp.cpu_count()) as p:
        return p.map(load_single_history, list(data_path.glob("history_*/")))


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: int = 0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (pad_size, 0)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class SequentialDataset(IterableDataset):
        def __init__(self, data_path: Path, seq_len: int = 40, reward_scale: float = 1.0,
                     masking_prob: float = 0., device: str = 'cpu'):

            with open(str(data_path/ "meta.json"), "r") as f:
                self.meta = json.load(f)

            self.state_dim = self.meta['state_dim']
            self.action_dim = self.meta['action_dim']

            self.reward_scale = reward_scale
            self.seq_len = seq_len
            print("Loading dataset...")
            self.histories = load_histories(data_path)
            print("Dataset loaded")
            self.masking_prob = masking_prob
            self.device = device

        def __prepare_random_sample(self):
            history_idx = np.random.randint(0, len(self.histories))
            hist = self.histories[history_idx]

            start_idx = np.random.randint(0, hist.shape[0])
            states = hist[0, start_idx:start_idx + self.seq_len]
            actions = hist[1, start_idx:start_idx + self.seq_len]
            returns = hist[2, start_idx:start_idx + self.seq_len, None]

            # pad up to seq_len if needed
            random_masking = np.int32(np.random.rand(states.shape[0], 1) > self.masking_prob)
            mask = np.vstack(
                [np.zeros((self.seq_len - states.shape[0], 1),), random_masking]
            )
            if states.shape[0] < self.seq_len:
                states = pad_along_axis(states, pad_to=self.seq_len)
                actions = pad_along_axis(actions, pad_to=self.seq_len)
                returns = pad_along_axis(returns, pad_to=self.seq_len)

            states = torch.from_numpy(states).type(torch.long)
            actions = torch.from_numpy(actions).type(torch.long)
            returns = torch.from_numpy(returns).type(torch.float32)
            mask = torch.from_numpy(mask).type(torch.bool)

            return states, actions, returns, mask

        def __iter__(self):
            while True:
                yield self.__prepare_random_sample()

