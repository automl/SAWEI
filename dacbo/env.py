from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import logging
from copy import deepcopy

import gym
import numpy as np
from ConfigSpace import Configuration
from smac.epm.utils import get_rng
from smac.facade.smac_ac_facade import SMAC4AC
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from target_algorithms.abstract_target_algorithm import AbstractTargetAlgorithm

from dacbo.utils.action_handling import action_translator
from dacbo.utils.io import configure_smac_loggers
from dacbo.utils.smac_interaction import (
    AcquisitionFunctions,
    build_smac_kwargs,
    get_acq_fun,
)
from dacbo.utils.target_algorithm_handling import get_target_algorithm

logging.basicConfig(level=logging.WARNING)


import random

import gym
import numpy as np
from gym.utils import seeding


class AbstractEnv(gym.Env):
    """
    Abstract template for environments
    """

    def __init__(self, config):
        """
        Initialize environment

        Parameters
        -------
        config : dict
            Environment configuration
            If to seed the action space as well
        """
        super(AbstractEnv, self).__init__()
        self.config = config
        if "instance_update_func" in self.config.keys():
            self.instance_updates = self.config["instance_update_func"]
        else:
            self.instance_updates = "round_robin"
        self.instance_set = config["instance_set"]
        self.instance_id_list = sorted(list(self.instance_set.keys()))
        self.instance_index = 0
        self.inst_id = self.instance_id_list[self.instance_index]
        self.instance = self.instance_set[self.inst_id]

        self.test = False
        if "test_set" in self.config.keys():
            self.test_set = config["test_set"]
            self.test_instance_id_list = sorted(list(self.test_set.keys()))
            self.test_instance_index = 0
            self.test_inst_id = self.test_instance_id_list[self.test_instance_index]
            self.test_instance = self.test_set[self.test_inst_id]

            self.training_set = self.instance_set
            self.training_id_list = self.instance_id_list
            self.training_inst_id = self.inst_id
            self.training_instance = self.instance
        else:
            self.test_set = None

        self.benchmark_info = config["benchmark_info"]
        self.initial_seed = None
        self.np_random = None

        self.n_steps = config["cutoff"]
        self.c_step = 0

        self.reward_range = config["reward_range"]

        if "observation_space" in config.keys():
            self.observation_space = config["observation_space"]
        else:
            if not config["observation_space_class"] == "Dict":
                try:
                    self.observation_space = getattr(gym.spaces, config["observation_space_class"])(
                        *config["observation_space_args"],
                        dtype=config["observation_space_type"],
                    )
                except KeyError:
                    print(
                        "Either submit a predefined gym.space 'observation_space' or an 'observation_space_class' as well as a list of 'observation_space_args' and the 'observation_space_type' in the configuration."
                    )
                    print("Tuple observation_spaces are currently not supported.")
                    raise KeyError

            else:
                try:
                    self.observation_space = getattr(gym.spaces, config["observation_space_class"])(
                        *config["observation_space_args"]
                    )
                except TypeError:
                    print(
                        "To use a Dict observation space, the 'observation_space_args' in the configuration should be a list containing a Dict of gym.Spaces"
                    )
                    raise TypeError

        # TODO: use dicts by default for actions and observations
        # The config could change this for RL purposes
        if "config_space" in config.keys():
            actions = config["config_space"].get_hyperparameters()
            action_types = [type(a).__name__ for a in actions]

            # Uniform action space
            if all(t == action_types[0] for t in action_types):
                if "Float" in action_types[0]:
                    low = np.array([a.lower for a in actions])
                    high = np.array([a.upper for a in actions])
                    self.action_space = gym.spaces.Box(low=low, high=high)
                elif "Integer" in action_types[0] or "Categorical" in action_types[0]:
                    if len(action_types) == 1:
                        try:
                            n = actions[0].upper - actions[0].lower
                        except:
                            n = len(actions[0].choices)
                        self.action_space = gym.spaces.Discrete(n)
                    else:
                        ns = []
                        for a in actions:
                            try:
                                ns.append(a.upper - a.lower)
                            except:
                                ns.append(len(a.choices))
                        self.action_space = gym.spaces.MultiDiscrete(np.array(ns))
                else:
                    raise ValueError("Only float, integer and categorical hyperparameters are supported as of now")
            # Mixed action space
            # TODO: implement this
            else:
                raise ValueError("Mixed type config spaces are currently not supported")
        elif "action_space" in config.keys():
            self.action_space = config["action_space"]
        else:
            try:
                self.action_space = getattr(gym.spaces, config["action_space_class"])(*config["action_space_args"])
            except KeyError:
                print(
                    "Either submit a predefined gym.space 'action_space' or an 'action_space_class' as well as a list of 'action_space_args' in the configuration"
                )
                raise KeyError

            except TypeError:
                print("Tuple and Dict action spaces are currently not supported")
                raise TypeError

        # seeding the environment after initialising action space
        self.seed(config.get("seed", None), config.get("seed_action_space", False))

    def step_(self):
        """
        Pre-step function for step count and cutoff

        Returns
        -------
        bool
            End of episode
        """
        done = False
        self.c_step += 1
        if self.c_step >= self.n_steps:
            done = True
        return done

    def reset_(self, instance=None, instance_id=None, scheme=None):
        """
        Pre-reset function for progressing through the instance set
        Will either use round robin, random or no progression scheme
        """
        self.c_step = 0
        if scheme is None:
            scheme = self.instance_updates
        self.use_next_instance(instance, instance_id, scheme=scheme)

    def use_next_instance(self, instance=None, instance_id=None, scheme=None):
        """
        Changes instance according to chosen instance progession

        Parameters
        -------
        instance
            Instance specification for potentional new instances
        instance_id
            ID of the instance to switch to
        scheme
            Update scheme for this progression step (either round robin, random or no progression)
        """
        if instance is not None:
            self.instance = instance
        elif instance_id is not None:
            self.inst_id = instance_id
            self.instance = self.instance_set[self.inst_id]
        elif scheme == "round_robin":
            self.instance_index = (self.instance_index + 1) % len(self.instance_id_list)
            self.inst_id = self.instance_id_list[self.instance_index]
            self.instance = self.instance_set[self.inst_id]
        elif scheme == "random":
            self.inst_id = np.random.choice(self.instance_id_list)
            self.instance = self.instance_set[self.inst_id]

    def step(self, action):
        """
        Execute environment step

        Parameters
        -------
        action
            Action to take

        Returns
        -------
        state
            Environment state
        reward
            Environment reward
        done : bool
            Run finished flag
        info : dict
            Additional metainfo
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset environment

        Returns
        -------
        state
            Environment state
        """
        raise NotImplementedError

    def get_inst_id(self):
        """
        Return instance ID

        Returns
        -------
        int
            ID of current instance
        """
        return self.inst_id

    def get_instance_set(self):
        """
        Return instance set

        Returns
        -------
        list
            List of instances

        """
        return self.instance_set

    def get_instance(self):
        """
        Return current instance

        Returns
        -------
        type flexible
            Currently used instance
        """
        return self.instance

    def set_inst_id(self, inst_id):
        """
        Change current instance ID

        Parameters
        ----------
        inst_id : int
            New instance index
        """
        self.inst_id = inst_id
        self.instance_index = self.instance_id_list.index(self.inst_id)

    def set_instance_set(self, inst_set):
        """
        Change instance set

        Parameters
        ----------
        inst_set: list
            New instance set
        """
        self.instance_set = inst_set
        self.instance_id_list = sorted(list(self.instance_set.keys()))

    def set_instance(self, instance):
        """
        Change currently used instance

        Parameters
        ----------
        instance:
            New instance
        """
        self.instance = instance

    def seed_action_space(self, seed=None):
        """
        Seeds the action space.
        Parameters
        ----------
        seed : int, default None
            if None self.initial_seed is be used

        Returns
        -------

        """
        if seed is None:
            seed = self.initial_seed

        self.action_space.seed(seed)

    def seed(self, seed=None, seed_action_space=False):
        """
        Set rng seed

        Parameters
        ----------
        seed:
            seed for rng
        seed_action_space: bool, default False
            if to seed the action space as well
        """

        self.initial_seed = seed
        # maybe one should use the seed generated by seeding.np_random(seed) but it can be to large see issue https://github.com/openai/gym/issues/2210
        random.seed(seed)
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        # uses the uncorrelated seed from seeding but makes sure that no randomness is introduces.

        if seed_action_space:
            self.seed_action_space()

        return [seed]

    def use_test_set(self):
        """
        Change to test instance set
        """
        if self.test_set is None:
            raise ValueError("No test set was provided, please check your benchmark config.")

        self.test = True
        self.training_set = self.instance_set
        self.training_id_list = self.instance_id_list
        self.training_inst_id = self.inst_id
        self.training_instance = self.instance

        self.instance_set = self.test_set
        self.instance_id_list = self.test_instance_id_list
        self.inst_id = self.test_inst_id
        self.instance = self.test_instance

    def use_training_set(self):
        """
        Change to training instance set
        """
        self.test = False
        self.test_set = self.instance_set
        self.test_instance_id_list = self.instance_id_list
        self.test_inst_id = self.inst_id
        self.test_instance = self.instance

        self.instance_set = self.training_set
        self.instance_id_list = self.training_id_list
        self.inst_id = self.training_inst_id
        self.instance = self.training_instance


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

    kwargs = {
        "low": dtype(np.array(lowers)),
        "high": dtype(np.array(uppers)),
        "dtype": dtype,
    }
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
            log_regret = np.log10(regret)
            reward = -log_regret  # we want to MINIMIZE regret and MAXIMIZE reward
    elif reward_type == "incumbent_value":
        reward = smac.get_runhistory().get_cost(smac.solver.incumbent)
    else:
        raise NotImplementedError

    return reward


class DACBOEnv(AbstractEnv):
    def __init__(
        self,
        instance_set: Dict,
        cutoff: int,
        action_space: gym.Space,
        seed: Optional[int] = None,
        reward_type: str = "log_regret",
        benchmark_info: Optional[str] = None,
        observation_types: Optional[List[str]] = None,
        normalize_observations: bool = True,
        smac_outdir: Optional[str] = None,
        initial_budget: Optional[int] = None,
    ):
        if reward_type == "log_regret":
            reward_range = [-np.inf, np.inf]
        elif reward_type == "incumbent_value":
            reward_range = [-np.inf, np.inf]
        else:
            raise ValueError(f"Unknown reward type {reward_type}.")
        self.reward_type = reward_type

        if smac_outdir is None:
            smac_outdir = None
        self.smac_outdir = smac_outdir
        self.initial_budget = initial_budget

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
        # self._smac_rngs: Optional[Dict[str, np.random.RandomState]] = None
        self._smac_rngs_state: Optional[Dict[str, Tuple]] = None
        self.target_algorithm: Optional[AbstractTargetAlgorithm] = None
        _, self.smac_rng = get_rng(rng=seed)  # : Optional[np.random.RandomState] = None

    def reset(self):
        self.reset_()  # selects next instance

        self.smac = None
        # self.smac_state = {
        #     "runhistory": None,
        #     "stats": None,
        #     "restore_incumbent": None
        # }
        self._smac_runhistory = None
        self._smac_stats = None
        self._smac_incumbent = None

        state = self.get_state()
        return state

    @abstractmethod
    def get_acquisition_function(self, action: Any) -> Tuple[AbstractAcquisitionFunction, Dict[str, Any]]:
        pass

    def build_smac_kwargs(self, action: Any):
        (
            acquisition_function,
            acquisition_function_kwargs,
        ) = self.get_acquisition_function(action=action)

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
            initial_budget=self.initial_budget,
        )
        return smac_kwargs

    def update_smac_state(self, smac: SMAC4AC):
        self._smac_runhistory = deepcopy(smac.runhistory)
        self._smac_stats = deepcopy(smac.stats)
        self._smac_incumbent = deepcopy(smac.solver.incumbent)

        # self._smac_rngs = {
        #     "smac.solver.config_space.random": deepcopy(smac.solver.config_space.random),
        #     "smac.solver.epm_chooser.model.rng": deepcopy(smac.solver.epm_chooser.model.rng)
        # }
        self._smac_rngs_state = {
            "smac.solver.config_space.random": smac.solver.config_space.random.get_state(),
            "smac.solver.epm_chooser.model.rng": smac.solver.epm_chooser.model.rng.get_state(),
        }
        # self.smac_rng = deepcopy(smac.solver.rng)
        # self.smac_state = {
        #     "runhistory": smac.runhistory,
        #     "stats": smac.stats,
        #     "restore_incumbent": smac.solver.incumbent  # smac.get_trajectory()[-1].incumbent
        # }

    def set_rngs(self, smac: SMAC4AC):
        pass
        # if self._smac_rngs is not None:
        #     for k, v in self._smac_rngs.items():
        #         key = k.replace("smac.", "")
        #         setattr(smac, key, v)
        #
        if self._smac_rngs_state is not None:
            smac.solver.config_space.random.set_state(self._smac_rngs_state["smac.solver.config_space.random"])
            smac.solver.epm_chooser.model.rng.set_state(self._smac_rngs_state["smac.solver.epm_chooser.model.rng"])

    def step(self, action):
        done = super(DACBOEnv, self).step_()

        # Change SMAC HP with action
        smac_kwargs = self.build_smac_kwargs(action=action)
        smac = SMAC4AC(**smac_kwargs)
        self.set_rngs(smac=smac)
        configure_smac_loggers(level=logging.INFO, detach=False)

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
        # print(list(smac.get_runhistory()))
        config_id = list(smac.get_runhistory())[-1].config_id
        configuration = list(smac.get_runhistory().ids_config[config_id].values())
        info = {
            "instance_id": self.inst_id,
            "cost": cost,
            "configuration": configuration,
            "runhistory": smac.get_runhistory(),
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


class DACBOAcqEnv(DACBOEnv):
    def __init__(
        self,
        instance_set: Dict,
        cutoff: int,
        seed: Optional[int] = None,
        action_values: Optional[List[str]] = None,
        reward_type: str = "log_regret",
        benchmark_info: Optional[str] = None,
        observation_types: Optional[List[str]] = None,
        normalize_observations: bool = True,
        smac_outdir: Optional[str] = None,
        initial_budget: Optional[int] = None,
    ):
        # Action Space
        if action_values is None:
            action_values = ["u_EI", "u_PI"]
        self.action_values = action_values
        action_space = gym.spaces.Discrete(n=len(self.action_values), seed=seed)

        super().__init__(
            instance_set=instance_set,
            cutoff=cutoff,
            seed=seed,
            action_space=action_space,
            reward_type=reward_type,
            benchmark_info=benchmark_info,
            observation_types=observation_types,
            normalize_observations=normalize_observations,
            smac_outdir=smac_outdir,
            initial_budget=initial_budget,
        )

    def get_acquisition_function(self, action: Any) -> Tuple[AbstractAcquisitionFunction, Dict[str, Any]]:
        action_value = self.action_values[action]

        if action_value.startswith("u"):
            action_id = action_translator(action_id=action_value)
            acquisition_function, acquisition_function_kwargs = get_acq_fun(acquisition_function_id=action_id)
        else:
            raise ValueError(f"Action must start with 'u'. Action: {action}.")

        return acquisition_function, acquisition_function_kwargs


class DACBOEIParEnv(DACBOEnv):
    """
    Optimize exploration parameter xi of EI.
    """

    def __init__(
        self,
        instance_set: Dict,
        cutoff: int,
        seed: Optional[int] = None,
        action_kwargs: Optional[Dict[str, Any]] = None,
        reward_type: str = "log_regret",
        benchmark_info: Optional[str] = None,
        observation_types: Optional[List[str]] = None,
        normalize_observations: bool = True,
        smac_outdir: Optional[str] = None,
    ):
        # Action Space
        if action_kwargs is None:
            action_kwargs = dict(
                action_bounds=(0, 10),
                normalize=True,
                log=False,
                normalized_bounds=(-1, 1),
            )

        low, high = action_kwargs["action_bounds"]
        # if action_kwargs["log"]:
        #     low = np.log10(low)
        #     high = np.log10(high)
        if action_kwargs["normalize"]:
            low, high = action_kwargs["normalized_bounds"]
        action_space = gym.spaces.Box(low=low, high=high, shape=(1,))
        self.action_kwargs = action_kwargs

        super().__init__(
            instance_set=instance_set,
            cutoff=cutoff,
            seed=seed,
            action_space=action_space,
            reward_type=reward_type,
            benchmark_info=benchmark_info,
            observation_types=observation_types,
            normalize_observations=normalize_observations,
            smac_outdir=smac_outdir,
        )

    def get_acquisition_function(self, action: float) -> Tuple[AbstractAcquisitionFunction, Dict[str, Any]]:
        acquisition_function, acquisition_function_kwargs = get_acq_fun(acquisition_function_id=AcquisitionFunctions.EI)

        if self.action_kwargs["normalize"]:
            # denormalize
            norm_bounds = self.action_kwargs["normalized_bounds"]
            target_bounds = self.action_kwargs["action_bounds"]
            norm_range = norm_bounds[1] - norm_bounds[0]
            target_range = target_bounds[1] - target_bounds[0]
            action = (action - (norm_bounds[0] - target_bounds[0])) * (target_range / norm_range)

        acquisition_function_kwargs["par"] = action

        return acquisition_function, acquisition_function_kwargs
