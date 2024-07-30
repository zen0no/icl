import wandb
import multiprocessing as mp

from typing import List, Dict
from numbers import Number


def log_run(project: str, group: str, name: str, config: dict, data: Dict[str, List[Number]]):
    run = wandb.init(project=project,
               group=group,
               name=name,
               config=config,)

    for t in zip(*data.values()):
        wandb.log(dict(zip(data, t)))

    run.finish()


def log_wandb_runs(project: str, group: str, name: str, config: dict, data: List[Dict[str, List[Number]]]):
    with mp.Pool(mp.cpu_count()) as p:
        p.starmap(log_run, [(project, group, name, config, run_data) for run_data in data])
