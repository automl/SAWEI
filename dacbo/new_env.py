from __future__ import annotations

from abc import abstractmethod
from typing import Any

import logging

import gym
import numpy as np
import smac
from hydra.utils import get_class, instantiate
from rich import print as printr
from smac import Scenario
from smac.callback import Callback
from smac.facade.abstract_facade import AbstractFacade
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.runhistory import StatusType, TrialInfo, TrialValue
from smac.utils.logging import get_logger

from dacbo.abstract_env import AbstractEnv
from dacbo.actions import EIXiActionAdmin, WEIActionAdmin
from dacbo.instances import TargetInstance
from dacbo.observations import Observer
from dacbo.rewards import RewardShaper

logger = get_logger(__name__)


class StepCallback(Callback):
    def __init__(self, budget_doe: int) -> None:
        super().__init__()
        self.trials_counter: int = 0
        self.budget_doe = budget_doe
        self._last_trial: tuple[TrialInfo, TrialValue]
        self.seen_trial: bool = False

    @property
    def last_trial(self) -> tuple[TrialInfo, TrialValue]:
        return self._last_trial

    @last_trial.setter
    def last_trial(self, trial: tuple[TrialInfo, TrialValue]) -> None:
        self._last_trial = trial
        self.seen_trial = True

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        # Leave optimization loop after one new evaluated config
        self.trials_counter += 1
        stay_in_loop = False

        # Stay in loop
        # - until the initial design was evaluated (no actions can be applied here)
        # - after exhausting budget to finish logging etc
        # Assumption: Everything
        if smbo.remaining_trials == 0 or self.trials_counter <= self.budget_doe:
            stay_in_loop = True

        self.last_trial = (info, value)

        return stay_in_loop

    def on_start(self, smbo: smac.main.smbo.SMBO) -> None:
        # Reset stop condition if we reenter the optimization
        smbo._stop = False


class DACBOEnv(AbstractEnv):
    def __init__(
        self,
        instance_set: dict[Any, TargetInstance],
        cutoff: int,
        action_admin_kwargs: dict | None,
        smac_kwargs: dict = {},
        smac_class: AbstractFacade | str = BlackBoxFacade,
        seed: int | None = None,
        reward_type: str = "log_regret",
        benchmark_info: str | None = None,
        observation_types: list[str] | None = None,
        normalize_observations: bool = True,
    ):
        # Reward function
        self.rewarder = RewardShaper(reward_type=reward_type)
        reward_range = self.rewarder.get_reward_range()

        # Observation Space
        self.observer = Observer(
            observation_types=observation_types, normalize_observations=normalize_observations, seed=seed
        )
        observation_space = self.observer.get_observation_space()

        # Action Space
        if action_admin_kwargs is None:
            self.action_admin = EIXiActionAdmin()
        else:
            name = action_admin_kwargs.get("class_name", "dacbo.new_env.EIXiActionAdmin")
            if "class_name" in action_admin_kwargs:
                del action_admin_kwargs["class_name"]
            cls = get_class(name)
            self.action_admin = cls(**action_admin_kwargs)
        action_space = self.action_admin.get_action_space()

        dacenv_config = {
            "instance_set": instance_set,
            "benchmark_info": benchmark_info,
            "cutoff": cutoff,
            "reward_range": reward_range,
            "observation_space": observation_space,
            "action_space": action_space,
            "seed": seed,
            "seed_action_space": None,
        }
        super().__init__(config=dacenv_config)

        # SMAC
        self.solver: AbstractFacade
        if type(smac_class) == str:
            smac_class = get_class(smac_class)
        self.smac_class = smac_class
        self.smac_kwargs = smac_kwargs

        self.budget_doe: int

    def build_solver(self):
        smac_kwargs = self.smac_kwargs.copy()

        # Build scenario
        scenario_kwargs = smac_kwargs["scenario"]
        scenario_kwargs["configspace"] = self.instance.configuration_space
        scenario = Scenario(**scenario_kwargs)
        smac_kwargs["scenario"] = scenario

        # Build initial design
        initial_design_kwargs = smac_kwargs["initial_design"]
        initial_design_kwargs["scenario"] = scenario
        self.budget_doe = initial_design_kwargs["n_configs"]
        initial_design = SobolInitialDesign(**initial_design_kwargs)
        smac_kwargs["initial_design"] = initial_design

        smac_kwargs["target_function"] = self.instance.target_function

        # Build acquisition function
        acq_fun_kwargs = smac_kwargs.get("acquisition_function", None)
        if acq_fun_kwargs is not None:
            acquisition_function = instantiate(acq_fun_kwargs)
            smac_kwargs["acquisition_function"] = acquisition_function

        # Build acquisition function maximizer
        acq_fun_max_kwargs = smac_kwargs.get("acquisition_maximizer", None)
        if acq_fun_max_kwargs is not None:
            acq_fun_max_kwargs["configspace"] = scenario.configspace
            acquisition_function_maximizer = instantiate(acq_fun_max_kwargs)
            smac_kwargs["acquisition_maximizer"] = acquisition_function_maximizer

        # Build config selector
        config_selector_kwargs = smac_kwargs.get("config_selector", None)
        if config_selector_kwargs is not None:
            config_selector = instantiate(config_selector_kwargs, _partial_=True)(scenario=scenario)
            smac_kwargs["config_selector"] = config_selector

        # Build runhistory encoder
        rh_encoder_kwargs = smac_kwargs.get("runhistory_encoder", None)
        if rh_encoder_kwargs is not None:
            runhistory_encoder = instantiate(rh_encoder_kwargs, _partial_=True)(scenario=scenario)
            smac_kwargs["runhistory_encoder"] = runhistory_encoder

        callback_kwargs = smac_kwargs.get("callbacks", None)
        if callback_kwargs:
            callbacks = [instantiate(cb) for cb in callback_kwargs]
            smac_kwargs["callbacks"] = callbacks

        # TODO fix SMAC output dir: `./output_directory/name/seed` -> `./output_directory`

        # Overwrite results so we can write in the same output directory
        smac_kwargs["overwrite"] = True

        # Add components changer
        self.step_callback = StepCallback(budget_doe=self.budget_doe)
        smac_kwargs = self.add_smac_callbacks(smac_kwargs)

        self.solver = self.smac_class(**smac_kwargs)

    def add_smac_callbacks(self, smac_kwargs: dict) -> dict:
        if not "callbacks" in smac_kwargs:
            smac_kwargs["callbacks"] = []
        smac_kwargs["callbacks"].append(self.step_callback)

        # smac_kwargs = self._add_action_callbacks(smac_kwargs=smac_kwargs)
        return smac_kwargs

    @abstractmethod
    def _add_action_callbacks(self, smac_kwargs: dict) -> dict:
        """Add Callbacks to SMAC enabling change of components

        Must be implemented in child class depending on the action space.

        Parameters
        ----------
        smac_kwargs : dict
            SMAC keyword args

        Returns
        -------
        dict
            Modified kwargs with added callbacks
        """
        return smac_kwargs

    def reset(self, return_info: bool = False):
        self.reset_()  # selects next instance

        self.build_solver()

        state, info = self.observer.observe(solver=self.solver, dacenv=self)
        ret = state
        if return_info:
            ret = state, info
        return ret

    def step(self, action):
        done = super(DACBOEnv, self).step_()

        # Change SMAC HP with action
        self.solver = self.action_admin.modify_solver(solver=self.solver, action=action)

        # Step SMAC
        self.solver.optimize()

        # Calculate reward
        reward = self.rewarder.get_reward(solver=self.solver, y_optimum=self.instance.y_optimum)

        # Determine state
        next_state, info = self.observer.observe(solver=self.solver, dacenv=self)

        logger.info(f"Step, next state, reward, done, info: {self.c_step}, {next_state}, {reward}, {done}, {info}")

        return next_state, reward, done, info
