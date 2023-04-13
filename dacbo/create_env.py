from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import random
import string

import gym
import hydra
import ioh
import numpy as onp
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from dacbo.utils.typing import Vector


def make_dacbo_env(cfg: DictConfig):
    import dacbo.env
    from dacbo.utils.instance_handling import read_instance_set, select_instance
    from dacbo.utils.target_algorithm_handling import get_target_algorithm

    # suite = cocoex.Suite(**cfg.coco_suite_kwargs)
    # instance = suite.get_problem_by_function_dimension_instance(**cfg.coco_instance)
    instance = ioh.get_problem(
        fid=cfg.coco_instance.function,
        instance=cfg.coco_instance.instance,
        dimension=cfg.coco_instance.dimension,
        problem_type="BBOB",
    )
    instance_set = {0: instance}

    # instance_set_kwargs = cfg.dacbo.benchmark.instance_set_kwargs
    n_dims = cfg.coco_instance.dimension
    # subtract initial design because that does not count into the envs cutoff NO
    budget = cfg.dacbo.benchmark.get("budget_multiplier", 20) * n_dims  # - compute_n_initial_design(n_dims)
    seed = cfg.seed
    observation_types = cfg.dacbo.benchmark.observation_types
    initial_budget = cfg.dacbo.benchmark.get("initial_budget_multiplier", 3) * cfg.coco_instance.dimension
    # instance_set, instance_set_path = read_instance_set(instance_set_kwargs)
    env = getattr(dacbo.env, cfg.dacbo.envtype)(
        instance_set=instance_set,
        cutoff=budget,
        seed=seed,
        observation_types=observation_types,
        reward_type=cfg.dacbo.benchmark.reward_type,
        initial_budget=initial_budget,
    )
    env.spec = gym.envs.registration.EnvSpec(cfg.env)
    for wrapper in cfg.env_wrappers:
        env = hydra.utils.instantiate(wrapper, env)

    # env = coax.wrappers.TrainMonitor(
    #     env, name=name or cfg.algo, tensorboard_dir=tensorboard_dir)
    return env


def set_seed_everywhere(seed: int):
    onp.random.seed(seed)
    random.seed(seed)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def get_baseline_policy(policy_type: str, budget: int, seed: int, cfg: DictConfig | None = None):
    if policy_type == "static (EI)":
        pi = [0] * budget
    elif policy_type == "static (PI)":
        pi = [1] * budget
    elif policy_type.startswith("explore-exploit"):
        switch_percentage = cfg.schedule_kwargs.percentage_before_switch
        # 0: EI, 1: PI
        pi = [0] * int(budget * switch_percentage) + [1] * int(budget * (1 - switch_percentage))
    elif policy_type == "round robin":
        pi = [0, 1] * (budget // 2)
    elif policy_type == "random":
        rng = np.random.default_rng(seed=seed)
        pi = rng.choice([0, 1], size=budget)
    else:
        raise ValueError(f"Unknown policy type {policy_type}.")
    return pi


def evaluate(policy, env, n_eval_episodes, seed, policy_name: str, policy_id: str):
    data = []
    for i in range(n_eval_episodes):
        rollout_data = rollout(env=env, policy=policy)
        rollout_data = pd.DataFrame(rollout_data)
        rollout_data["episode"] = [i] * len(rollout_data)
        rollout_data["policy_name"] = [policy_name] * len(rollout_data)
        rollout_data["policy"] = [policy_id] * len(rollout_data)
        rollout_data["seed"] = [seed] * len(rollout_data)
        data.append(rollout_data)
    data = pd.concat(data)
    data.reset_index(inplace=True, drop=True)
    return data


class DummyPolicy(object):
    def __init__(self, policy: Vector):
        self.policy = policy
        self.i: int = 0
        self.pi_iter = iter(self.policy)

    def get_next_action(self):
        return next(self.pi_iter)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.get_next_action(), None

    def mean(self, state):
        return self.get_next_action()


def rollout(policy, env):
    if type(policy) == list or type(policy) == onp.ndarray:
        policy = DummyPolicy(policy=policy)

    S = []
    R = []
    I = []
    A = []
    FV = []
    X = []
    done = False
    s = env.reset()
    i = 0
    while not done:
        S.append(s)
        a = policy.mean(s)  # use mean for exploitation
        s, r, done, info = env.step(a)
        i += 1
        R.append(r)
        A.append(a)

        if type(info) == dict:
            I.append(info.get("instance_id", None))
            FV.append(info.get("cost", None))
            X.append(info.get("configuration", None))

    T = onp.arange(0, len(S))
    data = {
        "step": T,
        "state": S,
        "action": A,
        "reward": R,
        "instance": I,
        "cost": FV,
        "configuration": X,
    }

    runhistory = None
    if type(info) == dict:
        runhistory = info.get("runhistory", None)
    if runhistory is not None:
        rhdata = runhistory.data
        initial_design_configs = []
        for rkey, rval in rhdata.items():
            config = runhistory.ids_config[rkey.config_id]
            if config.origin != "Local Search":
                initial_design_configs.append(list(config.values()))
        data["initial_design"] = [initial_design_configs] * len(T)

    return data
