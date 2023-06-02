from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gym
import numpy as np
import smac
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class ActionAdmin(object):
    @abstractmethod
    def get_action_space(self) -> gym.Space:
        ...

    @abstractmethod
    def modify_solver(self, solver: smac.main.smbo.SMBO, action: Any) -> smac.main.smbo.SMBO:
        return solver


class WEIActionAdmin(ActionAdmin):
    def __init__(self, alpha: float = 0.5, discrete_actions: list[float] | None = None) -> None:
        super().__init__()

        self.action_bounds = (0.0, 1.0)
        self.discrete_actions = discrete_actions
        self.alpha = alpha

    def get_action_space(self) -> gym.spaces.Discrete | gym.spaces.Box:
        if self.discrete_actions is None:
            low, high = self.action_bounds
            action_space = gym.spaces.Box(low=low, high=high, shape=(1,))
        else:
            action_space = gym.spaces.Discrete(len(self.discrete_actions))

        return action_space

    def modify_solver(self, solver: smac.main.smbo.SMBO, action: int | float | None) -> smac.main.smbo.SMBO:
        if action is not None:
            if self.discrete_actions is not None:
                action = self.discrete_actions[action]
            action = float(action)

            if not (self.action_bounds[0] <= action <= self.action_bounds[1]):
                raise ValueError(f"Action (xi) is '{action}' but only is allowed in range '{self.action_bounds}'.")

            kwargs = {
                "eta": solver.runhistory.get_cost(solver.intensifier.get_incumbent()), 
                "alpha": action,
                "num_data": solver.runhistory.finished,
            }
            solver.intensifier.config_selector._acquisition_function._update(**kwargs)
        return solver


class EIXiActionAdmin(ActionAdmin):
    def __init__(self, discrete_actions: list[float] | None = None, scale_factor: float = 0.1) -> None:
        """Adjust exploration-exploitation parameter in Expected Improvement Acquisition

        SMAC EI based on Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. 2012.
        “Practical Bayesian Optimization of Machine Learning Algorithms.” arXiv [stat.ML]. arXiv. http://arxiv.org/abs/1206.2944.

        If xi is
            - 0: Classic EI, more focus und exploitation
            - the higer, the more exploration
        Xi is in the scale of the transformed objective values (normalized to min and max observed).

        Costs are normalized within min, max observed so the scale changes over time.
        Therefore adjusting xi should happen proportional to the normalization bounds.
        Actions should be within -1,1 and multiplied with a scale factor.
        This scale factor is set manually so far based on the max-min range.

        """
        super().__init__()
        self.action_bounds = (-1, 1)
        self.discrete_actions = discrete_actions
        self.scale_factor = scale_factor

    def get_action_space(self) -> gym.spaces.Discrete | gym.spaces.Box:
        if self.discrete_actions is None:
            low, high = self.action_bounds
            action_space = gym.spaces.Box(low=low, high=high, shape=(1,))
        else:
            action_space = gym.spaces.Discrete(len(self.discrete_actions))

        return action_space

    def modify_solver(self, solver: smac.main.smbo.SMBO, action: int | float) -> smac.main.smbo.SMBO:
        # TODO use method update_acquisition_function
        if self.discrete_actions is not None:
            action = self.discrete_actions[action]
        action = float(action)

        if not (self.action_bounds[0] <= action <= self.action_bounds[1]):
            raise ValueError(f"Action (xi) is '{action}' but only is allowed in range '{self.action_bounds}'.")

        objective_bounds = solver.runhistory.objective_bounds
        if len(objective_bounds) > 0:
            objective_bounds = objective_bounds[0]  # single objective case
            range = np.abs(objective_bounds[0] - objective_bounds[1])
        else:
            range = 1
        xi = self.scale_factor * range * action
        logger.info(
            f"objective bounds: {objective_bounds}, range: {range}, scale factor: {self.scale_factor}, xi before: {action}, xi after: {xi}"
        )

        kwargs = {"eta": solver.runhistory.get_cost(solver.intensifier.get_incumbent()), "xi": xi}
        solver.intensifier.config_selector._acquisition_function._update(**kwargs)

        return solver
