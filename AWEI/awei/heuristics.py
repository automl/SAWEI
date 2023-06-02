from __future__ import annotations

import smac
from smac.callback import Callback
from smac.runhistory import TrialInfo, TrialValue

from awei.abstract_policy import AbstractPolicy
from awei.custom_types import State, Action


class WEITurnUpDownPolicy(AbstractPolicy):
    directions = ["exploiting", "exploring", "auto"]
    def __init__(
        self,
        alpha: float = 0.5,
        delta: float = 0.05,
        direction: str = "exploiting",
    ) -> None:
        self.alpha = alpha
        self.delta = delta
        assert direction in self.directions
        if direction == "exploiting":
            self.sign = 1
        elif direction == "exploring":
            self.sign = -1
        elif direction == "auto":
            self.sign = None
        self.bounds : tuple(float)= (0., 1.)
        self.last_inc_count: int = 0
        super().__init__()

    def act(self, state: State) -> Action:
        if state["n_incumbent_changes"] > self.last_inc_count:
            self.last_inc_count = state["n_incumbent_changes"]

            if self.sign is None:
                exploring = state["wei_pi_term"] <= state["wei_ei_term"]
                # If attitude is
                # - exploring (exploring==True): increase alpha, change to exploiting
                # - exploiting (exploring==False): decrease alpha, change to exploring
                sign = 1 if exploring else -1
            else:
                sign = self.sign

            alpha = self.alpha + sign * self.delta

            # Bound alpha
            lb, ub = self.bounds
            self.alpha = max(lb, min(ub, alpha))

        return self.alpha


def gentr_fn(alist):
    while True:
        for j in alist:
            yield j


class GutmannSobesterPulseSchedule(AbstractPolicy):
    def __init__(self, weights: list[float] | None = None):
        self.weights: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
        if weights is not None:
            self.weights = weights
        self.weight_generator = gentr_fn(self.weights)

    def act(self, state: State) -> Action:
        action = next(self.weight_generator)
        return action


class PortfolioAllocationPolicy(AbstractPolicy):
    def __init__(self) -> None:
        super().__init__()

    def act(self, state: State) -> Action:
        return None

class StaticPolicy(AbstractPolicy):
    def __init__(self) -> None:
        super().__init__()

    def act(self, state: State) -> Action:
        return None

