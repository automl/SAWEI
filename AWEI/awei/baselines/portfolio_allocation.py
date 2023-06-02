from __future__ import annotations

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac.callback import Callback
from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
from smac.runhistory import TrialInfo, TrialValue
import smac
from typing import Any
import numpy as np
from scipy.special import softmax
from smac.utils.configspace import convert_configurations_to_array
from smac.acquisition.maximizer.abstract_acqusition_maximizer import AbstractAcquisitionMaximizer
from smac.acquisition.maximizer.local_and_random_search import LocalAndSortedRandomSearch
from rich import print as printr
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger
import pandas as pd
from hydra.utils import get_class, instantiate

logger = get_logger(__name__)


def instantiate_acq_funs(acquisition_functions: list[str] | list[dict]) -> list[AbstractAcquisitionFunction]:
    acq_funs = []
    for af in acquisition_functions:
        if type(af) == str:
            if "smac" not in af:
                af = f"smac.acquisition.function.{af}"
            af = get_class(af)()
        else:
            af = instantiate(af)
        acq_funs.append(af)
    return acq_funs

class MultiAcquisitionFunction(AbstractAcquisitionFunction):
    def __init__(
        self, 
        acquisition_functions: list[AbstractAcquisitionFunction], 
        eta: float = 0.5,  # TODO set default from paper
    ) -> None:
        super().__init__()

        self._acquisition_functions = acquisition_functions
        self._eta = eta  # Line 1
        self._gains = np.zeros_like(self._acquisition_functions)  # Line 2
        self.acq_idx: int = -1  # needs to be set externally  
    
    @property
    def model(self) -> AbstractModel | None:
        """Return the used surrogate model in the acquisition function."""
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        """Updates the surrogate model."""
        self._model = model
        for acq_fun in self._acquisition_functions:
            acq_fun.model = model

    def _update(self, **kwargs: Any) -> None:
        for acq_fun in self._acquisition_functions:
            acq_fun._update(**kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the acquisition value for a given point X. This function has to be overwritten
        in a derived class.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X.
        """
        
        # # Nominate points from each acquisition function
        # # Here, we get the acq values. In __call__ we select the actual configurations
        # acq_values_list = [acq._compute(X=X) for acq in self._acquisition_functions]  # Line 4

        # # Line 5
        # # Select acq function
        # probs = softmax(self._eta * self._gains)
        # idx = np.random.choice(np.arange(0, len(self._acquisition_functions)), p=probs)
        # acq_values = acq_values_list[idx]

        # # TODO log gains

        # self._idx = idx  # save to know which gain to update
        assert self.acq_idx != -1
        return self._acquisition_functions[self.acq_idx]._compute(X=X)


class GPHedge(LocalAndSortedRandomSearch):
    def __init__(
        self, 
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        seed: int = 0,
        eta: float = 0.5,  # TODO set default from paper
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
    ) -> None:
        super().__init__(
            acquisition_function=acquisition_function,
            configspace=configspace,
            challengers=challengers,
            seed=seed,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            local_search_iterations=local_search_iterations,
        )

        # self._acquisition_functions = acquisition_functions
        # self._macq = MultiAcquisitionFunction(acquisition_functions)
        self._eta = eta  # Line 1
        self._gains = None
        self._idx: int = -1
        self._next_configs_by_acq_value_list: list[list[tuple[float, Configuration]]] = []

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        self._random_search._acquisition_function = acquisition_function
        self._local_search._acquisition_function = acquisition_function

        assert type(acquisition_function) == MultiAcquisitionFunction
        self._gains = np.zeros_like(acquisition_function._acquisition_functions)  # Line 2

    @property
    def gains(self) -> np.array:
        return self._gains

    @gains.setter
    def gains(self, gains: np.array) -> None:
        self._gains = gains  

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        """Implement acquisition function maximization.

        In contrast to `maximize`, this method returns an iterable of tuples, consisting of the acquisition function
        value and the configuration. This allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previously evaluated configurations.
        n_points: int
            Number of points to be sampled.

        Returns
        -------
        challengers : list[tuple[float, Configuration]]
            A list consisting of tuples of acquisition_value and its configuration.
        """

        # Line 4: Get configs from each acquisition function
        next_configs_by_acq_value_list = []
        n_acq_funcs = len(self.acquisition_function._acquisition_functions)
        for i in range(n_acq_funcs):
            self.acquisition_function.acq_idx = i
            acq_fun = self.acquisition_function._acquisition_functions[self.acquisition_function.acq_idx]
            next_configs_by_acq_value = super()._maximize(previous_configs=previous_configs, n_points=n_points)
            next_configs_by_acq_value_list.append(next_configs_by_acq_value)
        self._next_configs_by_acq_value_list = next_configs_by_acq_value_list

        # Line 5: Select nominee with softmax probability
        modulated_gains = np.array(self._eta * self._gains, dtype=float)
        probs = softmax(modulated_gains)
        idx = np.random.choice(np.arange(0, len(self.acquisition_function._acquisition_functions)), p=probs)
        next_configs_by_acq_value = next_configs_by_acq_value_list[idx]

        return next_configs_by_acq_value


class PortfolioAllocation(Callback):
    """ PortfolioAllocation

    # TODO add docstring for PortfolioAllocation
    """
    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict[str, float]] = []

    def on_tell_end(self, smbo: smac.main.smbo.SMBO, info: TrialInfo, value: TrialValue) -> bool | None:
        """Called after the stats are updated and the trial is added to the runhistory. Optionally, returns false
        to gracefully stop the optimization.
        """
        # print(smbo._intensifier.config_selector._acquisition_maximizer._next_configs_by_acq_value)
        assert type(smbo._intensifier.config_selector._acquisition_maximizer) == GPHedge

        # Line 8: Receive rewards: Mean of surrogate model for each acq fun's proposed config
        # Get the proposed configurations by each individual acq function
        next_configs_by_acq_value_list = smbo._intensifier.config_selector._acquisition_maximizer._next_configs_by_acq_value_list
        configurations: list[tuple[float, Configuration]] = [next_configs_by_acq_value[0] for next_configs_by_acq_value in next_configs_by_acq_value_list]
        if len(configurations) > 0:
            # Select the first configuration (assuming we retrain after 1)
            # TODO Is it ok to assume to select the first config?
            configurations = [ac[1] for ac in configurations]
            X = convert_configurations_to_array(configurations)
            if len(X.shape) == 1:
                X = X[np.newaxis, :]
            mu, var = smbo._intensifier.config_selector._model.predict_marginalized(X=X)
            # Mu is a vector of the number of acq functions
            rewards = np.squeeze(mu)  # remove the extra dimension again

            # Line 9: Update gains by observed reward
            smbo._intensifier.config_selector._acquisition_maximizer.gains += rewards
            logger.debug(f"gains {smbo._intensifier.config_selector._acquisition_maximizer.gains}, Last idx {smbo._intensifier.config_selector._acquisition_function.acq_idx}")

            acq_choices = [type(acq_fun).__name__ for acq_fun in smbo._intensifier.config_selector._acquisition_function._acquisition_functions] 
            idx = smbo._intensifier.config_selector._acquisition_function.acq_idx

            self.history.append({
                "n_evaluated": smbo.runhistory.finished,
                "gains": smbo._intensifier.config_selector._acquisition_maximizer.gains,
                "rewards": rewards,
                "acquisition_function_idx": idx,
                "acquisition_functions": acq_choices,
                "incumbent": smbo._intensifier._incumbents[0].get_array(),
                "incumbent_costs": smbo._intensifier._trajectory[-1].costs,
                }
            )

    def on_end(self, smbo: smac.main.smbo.SMBO) -> None:
        # Write history
        df = pd.DataFrame(data=self.history)
        df.to_json("portfolio_allocation_history.json", orient="split", indent=2)
        return super().on_end(smbo)
