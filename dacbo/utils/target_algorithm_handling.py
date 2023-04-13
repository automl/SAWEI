from typing import Dict, Optional, Union

from pathlib import Path

from ConfigSpace import Configuration, ConfigurationSpace, Float
from target_algorithms.abstract_target_algorithm import AbstractTargetAlgorithm
from target_algorithms.registration import (
    TargetAlgorithms,
    instantiate_target_algorithm,
)


class COCOTargetAlgorithm(AbstractTargetAlgorithm):
    name = "coco"

    def __init__(self, coco_problem):  # this is an ioh problem instance now
        super().__init__(configuration_space_file=None)

        self.name = str(coco_problem)
        self.coco_problem = coco_problem
        lower_bounds = self.coco_problem.constraint.lb  # self.coco_problem.lower_bounds
        upper_bounds = self.coco_problem.constraint.ub  # self.coco_problem.upper_bounds
        n_dim = self.coco_problem.meta_data.n_variables  # self.coco_problem.dimension
        hps = [Float(name=f"x{i}", bounds=[lower_bounds[i], upper_bounds[i]]) for i in range(n_dim)]
        self.configuration_space = ConfigurationSpace()
        self.configuration_space.add_hyperparameters(hps)

    def target_algorithm(self, configuration: Configuration) -> float:
        input = list(configuration.get_dictionary().values())
        output = self.coco_problem(input)
        return output


def get_target_algorithm(instance: Dict) -> AbstractTargetAlgorithm:
    if type(instance).__module__.startswith("ioh"):
        target_algorithm = COCOTargetAlgorithm(coco_problem=instance)
    else:
        ta_name = instance["target_algorithm"]
        enum_entry = ta_name.split(".")[-1]
        ta_id = getattr(TargetAlgorithms, enum_entry)
        n_dim = instance["n_dim"]
        configuration_space_file = instance["configuration_space_file"]
        parameters = instance.get("parameters", None)
        target_algorithm = instantiate_target_algorithm(
            ta_id,
            n_dim=n_dim,
            configuration_space_file=configuration_space_file,
            parameters=parameters,
        )
    return target_algorithm


def get_tae_runner(target_algorithm: AbstractTargetAlgorithm):
    return target_algorithm.target_algorithm


def get_tae_cs(target_algorithm: AbstractTargetAlgorithm) -> ConfigurationSpace:
    return target_algorithm.configuration_space
