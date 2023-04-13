from __future__ import annotations

import numpy as np
import smac


class AbstractRewardShaper(object):
    def get_reward_range(self) -> tuple[float, float]:
        raise NotImplementedError

    def get_reward(self, *args, **kwargs) -> float | list[float]:
        raise NotImplementedError


class RewardShaper(AbstractRewardShaper):
    choices = ["log_regret", "incumbent_cost"]

    def __init__(self, reward_type: str = "log_regret") -> None:
        if reward_type not in self.choices:
            raise ValueError(f"Unknown reward type {reward_type}. Valid choices are {self.choices}")
        self.reward_type = reward_type

    def get_reward_range(self) -> tuple[float, float]:
        if self.reward_type == "log_regret":
            reward_range = [-np.inf, np.inf]
        elif self.reward_type == "incumbent_value":
            reward_range = [-np.inf, np.inf]
        else:
            raise NotImplementedError

        return reward_range

    def get_reward(
        self,
        solver: smac.main.smbo.SMBO,
        y_optimum: float | None = None,
    ):
        incumbent = solver.intensifier.get_incumbent()
        assert incumbent is not None
        cost_incumbent = solver.runhistory.get_cost(incumbent)

        if self.reward_type == "incumbent_cost":
            # SMAC minimizes objectives, but RL maximizes rewards, so multiply with -1
            reward = -cost_incumbent
        elif self.reward_type == "log_regret":
            reward = cost_incumbent
            if y_optimum is None:
                raise ValueError("In order to calculate the regret, the global minimum (y_optimum) must be provided.")
            minimum = y_optimum
            regret = reward - minimum  # by definition this can't be lower than 0
            if np.isclose(regret, 0):
                reward = -np.inf  # TODO what to return if regret is close to 0?
            else:
                regret = np.abs(regret)  # but be safe ;)
                log_regret = np.log10(regret)
                reward = -log_regret  # we want to MINIMIZE regret and MAXIMIZE reward
        else:
            raise ValueError(f"Unknown reward type {self.reward_type}. Valid choices are {self.choices}")

        return reward
