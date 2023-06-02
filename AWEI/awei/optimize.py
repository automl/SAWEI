from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import shutil

import hydra
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from rich.logging import RichHandler

from dacbo.instances import create_instance_set
from dacbo.new_env import DACBOEnv

from awei.make_policy import make_policy
from awei.rollout import evaluate

import dacbo.utils.config_setup
import awei.utils.config_setup

logging.basicConfig(handlers=[RichHandler(markup=True)])
logger = logging.getLogger("optimize")


def add_meta_data(data: pd.DataFrame | pd.Series, meta_data: dict) -> pd.DataFrame | pd.Series:
    if type(data) == pd.DataFrame:
        for k, v in meta_data.items():
            data[k] = [v] * len(data)
    else:
        for k, v in meta_data.items():
            data[k] = v

    return data


def delete_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


@hydra.main("configs", "base", version_base="1.1")
def main(cfg: DictConfig):
    # if cfg.outdir == "awei_runs_debug":
    #     # Delete old debug runs

    if cfg.debug:
        delete_folder_contents(".")

    cfg.smac_kwargs.logging_level = instantiate(cfg.smac_kwargs.logging_level)

    dict_cfg = OmegaConf.to_container(cfg=cfg, resolve=True)
    logger.info(dict_cfg)
    printr(dict_cfg)
    # Instance set creation
    instance_set = create_instance_set(cfg)
    logger.info(instance_set)
    # printr("Instance set")
    # for k, v in instance_set.items():
    #     printr("\t", k, v)

    if cfg.schedule_id == "unknown":
        raise ValueError(f"schedule_id {cfg.schedule_id} is 'unknown'. Did you forget to set it?")

    env = DACBOEnv(
        instance_set=instance_set,
        cutoff=cfg.cutoff,
        smac_class=cfg.smac_class,
        smac_kwargs=OmegaConf.to_container(cfg=cfg.smac_kwargs, resolve=True),
        seed=cfg.seed,
        observation_types=cfg.observer.observation_types,
        normalize_observations=cfg.observer.normalize_observations,
        action_admin_kwargs=OmegaConf.to_container(cfg=cfg.action_admin, resolve=True),
    )

    policy = make_policy(cfg=cfg)
    state = env.reset()

    data = evaluate(
        env=env, policy=policy, n_eval_episodes=cfg.n_eval_episodes, seed=cfg.seed, policy_name=cfg.schedule_id
    )

    initial_design = pd.Series({"initial_design": data["initial_design"].iloc[-1]})
    del data["initial_design"]

    # Log
    data_fn = "rollout_data.json"
    initdesign_fn = "initial_design.json"

    data.to_json(data_fn, indent=2, orient="split")
    initial_design.to_json(initdesign_fn, indent=2, orient="split")

    # Delete smac output folder
    if not cfg.debug:
        shutil.rmtree("smac3_output")

    return data


if __name__ == "__main__":
    main()
