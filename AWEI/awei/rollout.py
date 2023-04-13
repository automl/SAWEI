from __future__ import annotations

import numpy as np
import pandas as pd
from awei.abstract_policy import AbstractPolicy
from awei.baseline import Schedule
from ConfigSpace import Configuration
from smac.runhistory import RunHistory

from dacbo.new_env import DACBOEnv


def rollout(policy: AbstractPolicy, env: DACBOEnv):
    if type(policy) == list or type(policy) == np.ndarray:
        policy = Schedule(policy=policy)

    if type(policy) == Schedule:
        policy.reset()

    S = []
    R = []
    I = []
    A = []
    FV = []
    X = []
    done = False
    s, info = env.reset(return_info=True)
    i = 0
    while not done:
        S.append(s)
        a = policy.act(s)  # use mean for exploitation
        s, r, done, info = env.step(a)
        i += 1
        R.append(r)
        A.append(a)

        if type(info) == dict:
            I.append(info.get("instance_id", None))
            FV.append(info.get("cost", None))
            config: Configuration | None = info.get("configuration", None)
            X.append(config.get_array())

    T = np.arange(0, len(S))
    data = {
        "step": T,
        "state": S,
        "action": A,
        "reward": R,
        "instance": I,
        "cost": FV,
        "configuration": X,
    }

    runhistory: RunHistory | None = None
    if type(info) == dict:
        runhistory = info.get("runhistory", None)
    if runhistory is not None:
        configs = runhistory.get_configs()
        initial_design_configs = [list(c.values()) for c in configs if "Initial Design" in c.origin]
        data["initial_design"] = [initial_design_configs] * len(T)

    return data


def evaluate(policy: AbstractPolicy, env: DACBOEnv, n_eval_episodes: int, seed: int, policy_name: str) -> pd.DataFrame:
    data = []
    for i in range(n_eval_episodes):
        rollout_data = rollout(env=env, policy=policy)
        rollout_data = pd.DataFrame(rollout_data)
        rollout_data["episode"] = [i] * len(rollout_data)
        rollout_data["policy_name"] = [policy_name] * len(rollout_data)
        # rollout_data["policy"] = [policy_id] * len(rollout_data)
        rollout_data["seed"] = [seed] * len(rollout_data)
        data.append(rollout_data)
    data = pd.concat(data)
    data.reset_index(inplace=True, drop=True)
    return data
