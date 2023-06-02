from typing import Any, Dict, List, Optional

import logging
from copy import deepcopy

import gym
import numpy as np
from ConfigSpace import Configuration
from dacbench.abstract_env import AbstractEnv
from smac.epm.util_funcs import get_rng
from smac.facade.smac_ac_facade import SMAC4AC
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from target_algorithms.abstract_target_algorithm import AbstractTargetAlgorithm

from bo_configuration_space import action_translator
from restart_utils import build_smac_kwargs, get_acq_fun, get_target_algorithm

logging.basicConfig(level=logging.WARNING)


def get_observation_space_kwargs(observation_types: List[str], normalize_observations: bool = True, dtype=np.float32):
    valids = ["remaining_steps"]
    lowers, uppers = [], []
    for obs_t in observation_types:
        if obs_t == "remaining_steps":
            if normalize_observations:
                lowers.append(0.0)
                uppers.append(1.0)
            else:
                # would need to pass budget here
                raise NotImplementedError
        else:
            raise ValueError(f"Unknown observation type {obs_t}. Valid choices are {valids}.")

    kwargs = {"low": np.array(lowers), "high": np.array(uppers), "dtype": dtype}
    return kwargs


def get_reward(
    smac: SMAC4AC,
    reward_type: str = "log_regret",
    global_minimum: Optional[float] = None,
):
    if reward_type == "log_regret":
        reward = smac.get_runhistory().get_cost(smac.solver.incumbent)
        if global_minimum is None:
            raise ValueError("In order to calculate the regret, the global minimum must be provided.")
        rh = smac.get_runhistory()
        minimum = global_minimum
        regret = reward - minimum  # by definition this can't be lower than 0
        if np.isclose(regret, 0):
            reward = -np.inf
        else:
            regret = np.abs(regret)  # but be safe ;)
            log_regret = np.log(regret)
            reward = log_regret
    else:
        raise NotImplementedError

    return reward


class DACBOEnv(AbstractEnv):
    def __init__(
        self,
        instance_set: Dict,
        cutoff: int,
        seed: Optional[int] = None,
        reward_type: str = "log_regret",
        benchmark_info: Optional[str] = None,
        action_values: Optional[List[str]] = None,
        observation_types: Optional[List[str]] = None,
        normalize_observations: bool = True,
        smac_outdir: Optional[str] = None,
    ):
        if reward_type == "log_regret":
            reward_range = [-np.inf, np.inf]
        else:
            raise ValueError(f"Unknown reward type {reward_type}.")
        self.reward_type = reward_type

        if smac_outdir is None:
            smac_outdir = None
        self.smac_outdir = smac_outdir

        # Action Space
        if action_values is None:
            action_values = ["u_EI", "u_LCB"]
        self.action_values = action_values
        action_space = gym.spaces.Discrete(n=len(self.action_values), seed=seed)

        # Observation Space
        if observation_types is None:
            observation_types = ["remaining_steps"]
        self.observation_types = observation_types
        self.normalize_observations = normalize_observations
        observation_space_kwargs = get_observation_space_kwargs(
            observation_types=self.observation_types,
            normalize_observations=normalize_observations,
        )
        observation_space_kwargs["seed"] = seed
        observation_space = gym.spaces.Box(**observation_space_kwargs)

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

        self.smac: Optional[SMAC4AC] = None
        self._smac_runhistory: Optional[RunHistory] = None
        self._smac_stats: Optional[Stats] = None
        self._smac_incumbent: Optional[Configuration] = None
        self._smac_rngs: Optional[Dict[str, np.random.RandomState]] = None
        self.target_algorithm: Optional[AbstractTargetAlgorithm] = None
        _, self.smac_rng = get_rng(rng=seed)  # : Optional[np.random.RandomState] = None

    def reset(self):
        self.reset_()  # selects next instance

        self.smac = None
        self.smac_state = {"runhistory": None, "stats": None, "restore_incumbent": None}

        state = self.get_state()
        return state

    def build_smac_kwargs(self, action: int):
        action_value = self.action_values[action]

        if action_value.startswith("u"):
            action_id = action_translator(action_id=action_value)
            acquisition_function, acquisition_function_kwargs = get_acq_fun(acquisition_function_id=action_id)
        else:
            raise ValueError(f"Action must start with 'u'. Action: {action}.")

        # Build target algorithm
        self.target_algorithm = get_target_algorithm(self.instance)
        budget = self.c_step
        seed = self.initial_seed if not self.smac_rng else self.smac_rng
        smac_kwargs = build_smac_kwargs(
            budget=budget,
            target_algorithm=self.target_algorithm,
            seed=seed,
            acquisition_function=acquisition_function,
            acquisition_function_kwargs=acquisition_function_kwargs,
            smac_outdir=self.smac_outdir,
            runhistory=self._smac_runhistory,
            restore_incumbent=self._smac_incumbent,
            stats=self._smac_stats,
        )
        return smac_kwargs

    def update_smac_state(self, smac: SMAC4AC):
        self._smac_runhistory = deepcopy(smac.runhistory)
        self._smac_stats = deepcopy(smac.stats)
        self._smac_incumbent = deepcopy(smac.solver.incumbent)

        self._smac_rngs = {
            "smac.solver.config_space.random": deepcopy(smac.solver.config_space.random),
            "smac.solver.epm_chooser.model.rng": deepcopy(smac.solver.epm_chooser.model.rng),
        }
        # self.smac_rng = deepcopy(smac.solver.rng)
        # self.smac_state = {
        #     "runhistory": smac.runhistory,
        #     "stats": smac.stats,
        #     "restore_incumbent": smac.solver.incumbent  # smac.get_trajectory()[-1].incumbent
        # }

    def set_rngs(self, smac: SMAC4AC):
        if self._smac_rngs is not None:
            for k, v in self._smac_rngs.items():
                key = k.replace("smac.", "")
                setattr(smac, key, v)

    def step(self, action):
        done = super(DACBOEnv, self).step_()

        # Change SMAC HP with action
        smac_kwargs = self.build_smac_kwargs(action=action)
        smac = SMAC4AC(**smac_kwargs)
        self.set_rngs(smac=smac)

        # Optimize for 1 step with SMAC  # TODO optimize for several steps
        smac.optimize()
        self.update_smac_state(smac=smac)

        # Calculate reward
        reward = get_reward(
            smac=smac,
            reward_type=self.reward_type,
            global_minimum=getattr(self.target_algorithm, "global_minimum", None),
        )

        # Determine state
        next_state = self.get_state()

        # Info
        # cost at timestep t
        cost = list(smac.get_runhistory().data.values())[-1].cost
        info = {
            "instance_id": self.inst_id,
            "cost": cost,
        }

        return next_state, reward, done, info

    def get_state(self, dummy: Optional[Any] = None) -> np.array:
        state_fun_map = {
            "remaining_steps": self.get_state_remaining_steps,
        }
        next_state = [state_fun_map[s](dummy) for s in self.observation_types]
        next_state = np.concatenate(next_state)
        return next_state

    def get_state_remaining_steps(self, dummy: Optional[Any] = None) -> List[float]:
        remaining_steps = self.n_steps - self.c_step
        if self.normalize_observations:
            remaining_steps = remaining_steps / self.n_steps
        return [remaining_steps]
