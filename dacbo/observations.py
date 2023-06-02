from __future__ import annotations

import gym
import numpy as np
import smac

from dacbo.abstract_env import AbstractEnv

registered_observation_types = [
    "remaining_steps",
    "wei_alpha",
    "wei_pi_term",
    "wei_pi_pure_term",
    "wei_pi_mod_term",
    "wei_ei_term",
    "ubr",
    "ubr_min_ucb",
    "ubr_min_lcb",
    "n_incumbent_changes",
]


def get_observation_space_kwargs(observation_types: list[str], normalize_observations: bool = True, dtype=np.float32):
    lowers, uppers = [], []
    for obs_t in observation_types:
        if obs_t == "remaining_steps":
            if normalize_observations:
                lowers.append(0.0)
                uppers.append(1.0)
            else:
                # would need to pass budget here
                raise NotImplementedError
        elif obs_t == "wei_alpha":
            lowers.append(0.0)
            uppers.append(1.0)
        elif obs_t == "wei_pi_term":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "wei_pi_pure_term":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "wei_pi_mod_term":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "wei_ei_term":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "ubr":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "ubr_min_ucb":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "ubr_min_lcb":
            lowers.append(-np.inf)
            uppers.append(np.inf)
        elif obs_t == "n_incumbent_changes":
            lowers.append(0)
            uppers.append(np.inf)
        else:
            raise ValueError(f"Unknown observation type {obs_t}. Valid choices are {registered_observation_types}.")

    kwargs = {
        "low": dtype(np.array(lowers)),
        "high": dtype(np.array(uppers)),
        "dtype": dtype,
    }
    return kwargs


def query_callback(solver: smac.main.smbo.SMBO, callback_type: str, key: str) -> float:
    from rich import print as printr
    obs = None
    for callback in solver._callbacks:
        if type(callback).__name__ == callback_type:
            if callback.history:
                obs = callback.history[-1][key]
            else:
                obs = -np.inf  # FIXME: alpha can only take 0-1 so -np.inf is not too correct
            break
    if obs is None:
        raise ValueError(f"Couldn't find the {callback_type} callback.")

    return obs


class Observer(object):
    def __init__(
        self,
        observation_types: list[str],
        normalize_observations: bool = True,
        dtype=np.float32,
        seed: int | None = None,
    ) -> None:
        if observation_types is None:
            observation_types = ["remaining_steps"]
        else:
            set_AB = set(observation_types) | set(registered_observation_types)
            if len(set_AB) > len(registered_observation_types):
                raise ValueError(
                    f"Invalid observation types specified: '{set(observation_types) - set(registered_observation_types)}'. Valid choices are '{registered_observation_types}'."
                )
        self.observation_types = observation_types
        self.normalize_observations = normalize_observations
        self.dtype = dtype
        self.seed = seed

    def get_observation_space(self) -> gym.spaces.Space:
        observation_space_kwargs = get_observation_space_kwargs(
            observation_types=self.observation_types,
            normalize_observations=self.normalize_observations,
        )
        observation_space_kwargs["seed"] = self.seed
        observation_space = gym.spaces.Box(**observation_space_kwargs)  # TODO enable dict observation space
        return observation_space

    def observe(self, solver: smac.main.smbo.SMBO, dacenv: AbstractEnv) -> np.array:
        observation = {}
        for obs_t in self.observation_types:
            # Remaining budget
            if obs_t == "remaining_steps":
                remaining_steps = dacenv.n_steps - dacenv.c_step
                if self.normalize_observations:
                    remaining_steps = remaining_steps / dacenv.n_steps
                obs = remaining_steps

            elif obs_t == "wei_alpha":
                obs = query_callback(solver=solver, callback_type="WEITracker", key="alpha")
            elif obs_t == "wei_pi_term":
                obs = query_callback(solver=solver, callback_type="WEITracker", key="pi_term")
            elif obs_t == "wei_pi_mod_term":
                obs = query_callback(solver=solver, callback_type="WEITracker", key="pi_mod_term")
            elif obs_t == "wei_pi_pure_term":
                obs = query_callback(solver=solver, callback_type="WEITracker", key="pi_pure_term")
            elif obs_t == "wei_ei_term":
                obs = query_callback(solver=solver, callback_type="WEITracker", key="ei_term")
            elif obs_t == "ubr":
                obs = query_callback(solver=solver, callback_type="UpperBoundRegretCallback", key="ubr")
            elif obs_t == "ubr_min_ucb":
                obs = query_callback(solver=solver, callback_type="UpperBoundRegretCallback", key="min_ucb")
            elif obs_t == "ubr_min_lcb":
                obs = query_callback(solver=solver, callback_type="UpperBoundRegretCallback", key="min_lcb")
            elif obs_t == "n_incumbent_changes":
                obs = solver._intensifier._incumbents_changed
            else:
                raise ValueError(f"Unknown observation type '{obs_t}'. Valid choices are '{registered_observation_types}'.")
            observation[obs_t] = obs

        # TODO move observation flattening to wrapper. Return dict
        # observation_flat = []
        # for v in observation.values():
        #     if np.isscalar(v):
        #         v = [v]
        #     observation_flat.append(v)
        # observation_flat = np.concatenate(observation_flat)

        # Info
        # cost at timestep t
        info = {}
        if dacenv.step_callback.seen_trial:
            trial_info, trial_value = dacenv.step_callback.last_trial
            cost = trial_value.cost
            configuration = trial_info.config
            info.update({
                "instance_id": dacenv.inst_id,
                "cost": cost,
                "configuration": configuration,
                "runhistory": solver.runhistory,
            })

        return observation, info
