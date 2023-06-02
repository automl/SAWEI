from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from ConfigSpace.hyperparameters import FloatHyperparameter
from smac.configspace import (Configuration, ConfigurationSpace,
                              UniformFloatHyperparameter)

from target_algorithms.abstract_target_algorithm import AbstractTargetAlgorithm


class AbstractSyntheticFunction(AbstractTargetAlgorithm):
    @property
    @abstractmethod
    def n_dim_max(self) -> Optional[int]:
        """
        Maximum number of dimensions. If #dimensions is infinite, set n_dim_max to None.
        """

    @property
    @abstractmethod
    def global_minimum(self) -> float:
        """
        Value of global minimum.
        """

    @property
    @abstractmethod
    def x_optimal(self) -> Union[Union[List, np.array], List[Union[List, np.array]]]:
        """
        Optimal configuration(s).
        """

    def __init__(self, n_dim: int, configuration_space_file: Optional[Union[str, Path]] = None):
        check_n_dim(n_dim=n_dim, n_dim_max=self.n_dim_max)
        if self.global_minimum is None or self.x_optimal is None:
            raise ValueError("Please set global_minimum and x_optimal to sth other than None in child class.")

        super().__init__(configuration_space_file=configuration_space_file)

    @abstractmethod
    def function(self, x: np.array) -> float:
        """
        Function/equation goes here.

        Parameters
        ----------
        x : np.array of shape (n_dim,)

        Returns
        -------
        Function value.

        """

    def target_algorithm(self, configuration: Configuration) -> float:
        cfg = configuration.get_dictionary()
        vals = np.asarray(list(cfg.values()))
        ret = self.function(vals)
        return ret


def check_n_dim(n_dim, n_dim_max):
    if n_dim_max is not None:
        if n_dim > n_dim_max:
            raise ValueError(f"n_dim {n_dim} must be smaller equal n_dim_max {n_dim_max}.")
    if n_dim <= 0:
        raise ValueError(f"n_dim {n_dim} must be greater than 0.")


def expand_bound(bound: Union[np.number, np.array], n_dim: int) -> np.array:
    if np.isscalar(bound):
        bound = np.array([bound] * n_dim)
    return bound


def construct_configuration_space(
    n_dim: int,
    lower_bounds: Union[np.number, np.array],
    upper_bounds: Union[np.number, np.array],
    defaults: Optional[Union[np.number, np.array]] = None,
    hp_types: Optional[List[FloatHyperparameter]] = None,
    hp_default_type: FloatHyperparameter = UniformFloatHyperparameter,
    sample_logs: Optional[List[bool]] = None,
) -> ConfigurationSpace:
    lower_bounds = expand_bound(bound=lower_bounds, n_dim=n_dim)
    upper_bounds = expand_bound(bound=upper_bounds, n_dim=n_dim)
    if defaults is None:
        rng = np.random.default_rng(seed=789)
        defaults = rng.uniform(low=lower_bounds, high=upper_bounds)
    else:
        defaults = expand_bound(bound=defaults, n_dim=n_dim)
    if hp_types is None:
        hp_types = [hp_default_type] * n_dim
    if sample_logs is None:
        sample_logs = [False] * n_dim

    cs = ConfigurationSpace()
    hyperparameters = []
    for i in range(n_dim):
        hp_name = f"x{i}"
        lower_bound = lower_bounds[i]
        upper_bound = upper_bounds[i]
        default = defaults[i]
        hp_type = hp_types[i]
        samplelog = sample_logs[i]
        hp = hp_type(name=hp_name, lower=lower_bound, upper=upper_bound, default_value=default, log=samplelog)
        hyperparameters.append(hp)
    cs.add_hyperparameters(hyperparameters)

    return cs


class PyBenchFCNSOPFunction(AbstractSyntheticFunction):
    sop_class = None
    name = None
    n_dim_max = None
    global_minimum = None
    x_optimal = None

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None):
        # self.sop_class = sop_class
        self.fcn = self.sop_class(n_var=n_dim)
        global_minimum = self.fcn.optimalF
        if np.isclose(global_minimum, 0, atol=1e-15):
            global_minimum = 0.0
        self.global_minimum = global_minimum
        self.x_optimal = self.fcn.optimalX
        # self.function = self.fcn.f
        self.n_dim = n_dim

        self.fcn.boundaries = self.fcn.boundaries.astype(np.float)

        lower_bounds = self.fcn.boundaries[0]
        upper_bounds = self.fcn.boundaries[1]

        cs = construct_configuration_space(
            n_dim=self.n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self.configuration_space = cs

        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x: np.array) -> float:
        return self.fcn.f(x)


def pybenchfcn_sop_class_constructor(sop_class, name: Optional[str] = None):
    if name is None:
        fcn_name = sop_class.__name__
        fcn_name = fcn_name.replace("fcn", "")
        name = fcn_name.capitalize()

    inst = sop_class()
    if inst.n_var == 2:
        n_dim_max = 2
    else:
        n_dim_max = None

    cls_dict = dict(PyBenchFCNSOPFunction.__dict__)
    cls_dict["name"] = name
    cls_dict["sop_class"] = sop_class
    cls_dict["n_dim_max"] = n_dim_max

    synthfuncls = type(name, (PyBenchFCNSOPFunction,), cls_dict)

    return synthfuncls


if __name__ == "__main__":
    from PyBenchFCN.SingleObjectiveProblem import levin13fcn

    cls = pybenchfcn_sop_class_constructor(sop_class=levin13fcn, name="LeviN13")
    levin13 = cls()
