from typing import List, Optional, Tuple, Union

import numpy as np
from dacbench import AbstractEnv

from dacbo.utils.typing import Vector


class DummyPolicy(object):
    def __init__(self, policy: Vector):
        self.policy = policy
        self.i: int = 0
        self.pi_iter = iter(self.policy)

    def get_next_action(self):
        return next(self.pi_iter), None

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.get_next_action()


def rollout(
    env: AbstractEnv,
    policy: Union[Vector, "stable_baselines3.common.base_class.BaseAlgorithm"],
):
    if type(policy) == list or type(policy) == np.ndarray:
        policy = DummyPolicy(policy=policy)

    S = []
    R = []
    I = []
    A = []
    FV = []
    done = False
    s = env.reset()
    i = 0
    while not done:
        S.append(s)
        a, _ = policy.predict(observation=s)
        s, r, done, info = env.step(action=a)
        i += 1
        R.append(r)
        A.append(a)

        if type(info) == dict:
            I.append(info.get("instance_id", None))
            FV.append(info.get("cost", None))

    T = np.arange(0, len(S))
    data = {
        "step": T,
        "state": S,
        "action": A,
        "reward": R,
        "instance": I,
        "cost": FV,
    }
    return data
