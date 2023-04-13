from typing import Any

import ioh
from ConfigSpace import Configuration, ConfigurationSpace, Float
from ioh import ProblemType
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as printr


class TargetInstance(object):
    def __init__(
        self,
        target_function: callable,
        configuration_space: ConfigurationSpace,
        x_optimum: list[float] | None = None,
        y_optimum: float | None = None,
    ) -> None:
        self.target_function = target_function
        self.configuration_space = configuration_space
        self.x_optimum = x_optimum
        self.y_optimum = y_optimum

    def __call__(self, configuration: Configuration, seed: int | None = None) -> Any:
        return self.target_function(configuration=configuration, seed=seed)

    def __str__(self) -> str:
        rep = f"Target function: '{self.target_function}', configuration space: {self.configuration_space}"
        return rep


def create_instance_set(cfg: DictConfig) -> dict[int, Any]:
    """Create Instance Set

    How to specify an instance set:

    ```yaml
    benchmark: BBOB  # specify the benchmark name
    instance_set:
        - instance_kwargs1
        - instance_kwargs2
    ```
    `instance_set` can either be a list of dicts (several instances) or a dict (one instance).

    For the BBOB benchmark the instance kwargs are:
        fid, instance, dimension

    Parameters
    ----------
    cfg : DictConfig
        Configuration (hydra)

    Returns
    -------
    dict[int, Any]
        Instance set

    Raises
    ------
    NotImplementedError
        If the benchmark is not registered.
    """
    if cfg.benchmark == "BBOB":
        if type(cfg.instance_set) != ListConfig:
            cfg.instance_set = ListConfig(
                [
                    cfg.instance_set,
                ]
            )
        instance_set = {}
        for i, cfg_instance in enumerate(cfg.instance_set):
            # Problem instance
            problem = ioh.get_problem(
                fid=cfg_instance.fid,
                instance=cfg_instance.instance,
                dimension=cfg_instance.dimension,
                problem_type=ProblemType.BBOB,
            )

            # Configuration space
            lower_bounds = problem.bounds.lb
            upper_bounds = problem.bounds.ub
            n_dim = problem.meta_data.n_variables
            hps = [Float(name=f"x{i}", bounds=[lower_bounds[i], upper_bounds[i]]) for i in range(n_dim)]
            configuration_space = ConfigurationSpace(seed=cfg.seed)
            configuration_space.add_hyperparameters(hps)

            def target_function(configuration: Configuration, seed: int | None = None) -> float:
                input = list(configuration.get_dictionary().values())
                output = problem(input)
                return output

            instance = TargetInstance(
                target_function=target_function,
                configuration_space=configuration_space,
                x_optimum=problem.optimum.x,
                y_optimum=problem.optimum.y,
            )

            instance_set[i] = instance
    else:
        raise NotImplementedError(f"We only know the BBOB benchmark so far. Got '{cfg.benchmark}'.")

    return instance_set
