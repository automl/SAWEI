from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import trim_mean
import smac
from smac.acquisition.function import LCB, UCB
from smac.acquisition.maximizer import (
    AbstractAcquisitionMaximizer,
    LocalAndSortedRandomSearch,
)
from smac.callback import Callback
from smac.main.smbo import SMBO
from smac.model.random_forest import RandomForest
from smac.model.gaussian_process import GaussianProcess
from smac.runhistory import TrialInfo, TrialKey, TrialValue
from ConfigSpace import Configuration

from dacbo.weighted_expected_improvement import WEI, EIPI

from awei.abstract_policy import AbstractPolicy
from awei.custom_types import Action, State

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class UpperBoundRegretCallback(Callback):
    def __init__(self, top_p: float = 0.5) -> None:
        super().__init__()

        # Use only top p portion of the evaluated configs to fit the model (Sec. 4)
        # DISCUSS: What does top p configs mean for AC?
        self.top_p: float = top_p  
        self.ubr: float | None = None
        self.history: list[float] = []
        self._UCB: UCB = UCB()
        self._LCB: LCB = LCB()

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        # Line 16: r_t = min UCB(config) (from all evaluated configs) - min LCB(config) (from config space)
        # Get all evaluated configs
        rh = smbo.runhistory
        evaluated_configs = rh.get_configs(sort_by="cost")
        evaluated_configs = evaluated_configs[:int(np.ceil(len(evaluated_configs) * self.top_p))]

        # Prepare acquisition functions
        model = smbo.intensifier._config_selector._model
        # BUG: num data is calculated wrongly
        # calculate UBR right from the start, filter to sbo if necessary
        if (type(model) == GaussianProcess and model._is_trained) or (type(model) == RandomForest and model._rf is not None): 
            kwargs = {"model": model, "num_data": rh.finished}
            self._UCB.update(**kwargs)
            self._LCB.update(**kwargs)

            # Minimize UCB (max -UCB) for all evaluated configs
            acq_values = self._UCB(evaluated_configs)
            min_ucb =  -float(np.squeeze(np.amax(acq_values)))

            # Minimize LCB (max -LCB) on config space
            acq_maximizer = LocalAndSortedRandomSearch(
                configspace=smbo._configspace,
                seed=smbo.scenario.seed,
                acquisition_function=self._LCB,
            )
            challengers = acq_maximizer._maximize(
                evaluated_configs,
                n_points=1,
            )
            challengers = np.array(challengers, dtype=object)
            acq_values = challengers[:, 0]
            min_lcb = -float(np.squeeze(np.amax(acq_values)))

            self.ubr = min_ucb - min_lcb

            self.history.append({
                "n_evaluated": smbo.runhistory.finished,
                "ubr": self.ubr,
                "min_ucb": min_ucb,
                "min_lcb": min_lcb,
            })

        return super().on_tell_end(smbo, info, value)

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        # FIXME: Writes after each iteration bc of env
        # Write history
        df = pd.DataFrame(data=self.history)
        df.to_json("ubr_history.json", orient="split", indent=2)
        return super().on_end(smbo)


class WEITracker(Callback):
    def __init__(self) -> None:
        self.history: list[dict] = []
        super().__init__()

    def on_next_configurations_end(self, config_selector: smac.main.config_selector.ConfigSelector, config: Configuration) -> None:
        if type(config_selector._acquisition_function) in (WEI, EIPI) and \
            ((type(config_selector._model) == GaussianProcess and config_selector._model._is_trained) \
                or (type(config_selector._model) == RandomForest and config_selector._model._rf is not None)):  # FIXME: Flag _is_trained only exists for GP so far:
            X = config.get_array()
            # TODO: pipe X through
            acq_values = config_selector._acquisition_function([config])
            alpha = config_selector._acquisition_function._alpha
            pi_term = config_selector._acquisition_function.pi_term[0][0]
            ei_term = config_selector._acquisition_function.ei_term[0][0]

            self.history.append({
                "n_evaluated": config_selector._runhistory.finished,
                "alpha": alpha,
                "pi_term": pi_term,
                "ei_term": ei_term,
                "pi_pure_term": config_selector._acquisition_function.pi_pure_term[0][0],
                "pi_mod_term": config_selector._acquisition_function.pi_mod_term[0][0],
            })
        return super().on_next_configurations_end(config_selector, config)

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        # FIXME: Writes after each iteration bc of env
        # Write history
        df = pd.DataFrame(data=self.history)
        df.to_json("wei_history.json", orient="split", indent=2)
        return super().on_end(smbo)



def detect_switch(UBR: np.array, window_size: int = 10, atol_rel: float = 0.1) -> np.array[bool]:
    miqm = apply_moving_iqm(U=UBR, window_size=window_size)
    miqm_gradient = np.gradient(miqm)

    # max_grad = np.maximum.accumulate(miqm_gradient)
    # switch = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel*max_grad[i]) for i in range(len(miqm_gradient))])
    # switch[0] = 0  # misleading signal bc of iqm

    G_abs = np.abs(miqm_gradient)
    max_grad = [np.nanmax(G_abs[:i+1]) for i in range(len(G_abs))]
    switch = np.array([np.isclose(miqm_gradient[i], 0, atol=atol_rel*max_grad[i]) for i in range(len(miqm_gradient))])
    # switch = np.isclose(miqm_gradient, 0, atol=1e-5)
    switch[:window_size] = 0  # misleading signal bc of iqm
    
    return switch

# Moving IQM
def apply_moving_iqm(U: np.array, window_size: int = 5) -> np.array:

    def moving_iqm(X: np.array) -> float:
        return trim_mean(X, 0.25)

    U_padded = np.concatenate((np.array([U[0]] * (window_size - 1)), U))
    slices = sliding_window_view(U_padded, window_size)
    miqm = np.array([moving_iqm(s) for s in slices])
    return miqm


class AWEIPolicy(AbstractPolicy):
    def __init__(
        self, 
        alpha: float = 0.5, 
        delta: float | str = 0.1,
        window_size: int = 7,
        atol_rel: float = 0.1,
        track_attitude: str = "last",
        use_pure_PI: bool = False,
        auto_alpha: bool = False,
    ) -> None:
        # alpha = 1: PI = Exploiting
        # alpha = 0.5: EI = Exploring/Balance
        # alpha = 0: Exploring
        self.alpha = alpha
        self.delta = delta
        self.window_size = window_size
        self.atol_rel = atol_rel
        self.track_attitude = track_attitude
        self.use_pure_PI = use_pure_PI
        self.auto_alpha = auto_alpha

        if self.delta == "auto":
            self.auto_alpha = True

        self.last_inc_count: int = 0
        self._pi_term_sum: float = 0.
        self._ei_term_sum: float = 0.
        self.bounds = (0., 1.)
        self.S: list[State] = []

    def act(self, state: State) -> Action:
        self.S.append(state)
        # Check if it is time to switch
        switch = False
        UBR = [s["ubr"] for s in self.S]

        if self.use_pure_PI:
            key_pi = "wei_pi_pure_term"
        else:
            key_pi = "wei_pi_mod_term"

        # first observation is -np.inf bc we cannot calculate UBR yet,
        # and we need at least 2 UBRs to compute the gradient
        if len(UBR) > 2:  
            switch = detect_switch(UBR=UBR[1:], window_size=self.window_size, atol_rel=self.atol_rel)[-1]

        self._pi_term_sum += state[key_pi]
        self._ei_term_sum += state["wei_ei_term"]

        if switch:
            if self.track_attitude == "last":
                # Calculate attitude: Exploring or exploiting?
                # Exploring = when ei term is bigger
                # Exploiting = when pi term is bigger
                exploring = state[key_pi] <= state["wei_ei_term"]
                distance = state["wei_ei_term"] - state[key_pi]
            elif self.track_attitude in ["until_inc_change", "until_last_switch"]:
                exploring =  self._pi_term_sum <= self._ei_term_sum
                distance = self._ei_term_sum - self._pi_term_sum
            else:
                raise ValueError(f"Unknown track_attitude {self.track_attitude}.")

            if self.auto_alpha:
                alpha = sigmoid(distance)
            else:
                # If attitude is
                # - exploring (exploring==True): increase alpha, change to exploiting
                # - exploiting (exploring==False): decrease alpha, change to exploring
                sign = 1 if exploring else -1
                alpha = self.alpha + sign * self.delta

            # Bound alpha
            lb, ub = self.bounds
            self.alpha = max(lb, min(ub, alpha))

        if self.track_attitude == "until_inc_change":
            if state["n_incumbent_changes"] > self.last_inc_count:
                self.last_inc_count = state["n_incumbent_changes"]
                self._pi_term_sum: float = 0.
                self._ei_term_sum: float = 0.
        elif self.track_attitude == "until_last_switch":
            if switch:
                self._pi_term_sum: float = 0.
                self._ei_term_sum: float = 0.

        return self.alpha


