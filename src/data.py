import numpy as np

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
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


class SequentialDataset(IterableDataset):
        def __init__(self, data_path: Path, state_dim, action_dim, seq_len: int = 40, reward_scale: float = 1.0,
                     masking_prob=0.):
            self.reward_scale = reward_scale
            self.seq_len = seq_len
            self.histories = load_histories(data_path)

            self.state_dim = state_dim
            self.action_dim = action_dim

            self.masking_prob = masking_prob

        def __prepare_random_sample(self):
            history_idx = np.random.randint(0, len(self.histories))
            hist = self.histories[history_idx]

            start_idx = np.random.randint(0, hist.shape[0])
            states_idx = hist[0, start_idx:start_idx + self.seq_len]
            actions_idx = hist[2, start_idx:start_idx + self.seq_len]
            returns = hist[1, start_idx:start_idx + self.seq_len]
            time_steps = np.arange(start_idx, start_idx + self.seq_len)

            states = np.eye(self.state_dim)[states_idx, :]
            actions = np.eye(self.action_dim)[actions_idx, :]

            # pad up to seq_len if needed
            random_masking = np.int32(np.random.rand() > self.masking_prob)
            mask = np.hstack(
                [random_masking, np.zeros(self.seq_len - states.shape[0])]
            )
            if states.shape[0] < self.seq_len:
                states = pad_along_axis(states, pad_to=self.seq_len)
                actions = pad_along_axis(actions, pad_to=self.seq_len)
                returns = pad_along_axis(returns, pad_to=self.seq_len)

            return states, actions, returns, time_steps, mask

        def __iter__(self):
            while True:
                yield self.__prepare_random_sample()

