from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from smac.configspace import Configuration

from smacaux.smac_utils import build_configuration_space


class AbstractTargetAlgorithm(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Descriptive name of target algorithm.
        """

    def __init__(self, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        if configuration_space_file is not None:
            self.configuration_space = build_configuration_space(cs_file=configuration_space_file)

    @abstractmethod
    def target_algorithm(self, configuration: Configuration) -> float:
        """
        Evaluates the target algorithm/function for a given configuration. Called by SMAC.

        Parameters
        ----------
        configuration: Configuration
            Hyperparameter configuration.

        Returns
        -------
        Objective value. SMAC minimizes.

        """
