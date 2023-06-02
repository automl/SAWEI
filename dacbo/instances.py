from typing import Any

import ioh
from ConfigSpace import Configuration, ConfigurationSpace, Float
from ioh import ProblemType
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as printr
from functools import partial
import numpy as np


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
    if type(cfg.instance_set) != ListConfig:
        cfg.instance_set = ListConfig(
            [
                cfg.instance_set,
            ]
        )
    if cfg.benchmark == "BBOB":
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
    elif cfg.benchmark.startswith("HPOBench"):
        available_task_ids = {
            "rf"  : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168329, 168330, 168331, 168335, 168868, 168908, 168910, 168911, 168912],
            "xgb" : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168911, 168912],
            "svm" : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168329, 168330, 168331, 168335, 168868, 168908, 168909, 168910, 168911, 168912],
            "lr"  : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168329, 168330, 168331, 168335, 168868, 168908, 168909, 168910, 168911, 168912],
            "nn"  : [31, 53, 3917, 9952, 10101, 146818, 146821, 146822],
        }
        available_seeds = {
            "rf": [665, 1319, 7222, 7541, 8916],
            "xgb": [665, 1319, 7222, 7541, 8916],
            "svm": [665, 1319, 7222, 7541, 8916],
            "lr": [665, 1319, 7222, 7541, 8916],
            "nn": [665, 1319, 7222, 7541, 8916],
        }

        instance_set = {}
        for i, cfg_instance in enumerate(cfg.instance_set):
            from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark

            if cfg_instance.task_id not in available_task_ids[cfg_instance.model]:
                raise ValueError(f"Task id {cfg_instance.task_id} not in available tasks {available_task_ids[instance.model]}.")

            benchmark = TabularBenchmark(
                model=cfg_instance.model, 
                task_id=cfg_instance.task_id,
                rng=cfg.seed
            )

            seeds = benchmark._seeds_used()
            seeds.sort()
            # printr(benchmark.global_minimums)
            # printr(seeds)

            # if cfg_instance.seed not in seeds:
            #     raise ValueError(f"Instance {cfg_instance.seed} not in available seeds {seeds}.")
            
            rng = np.random.default_rng(seed=cfg.seed)

            if cfg.smac_kwargs.scenario.deterministic:
                def target_function(configuration: Configuration, seed: int | None = None, **kwargs) -> float:
                    outs = []
                    for seed in seeds:
                        output_dict = benchmark.objective_function(configuration=configuration, seed=seed, **kwargs)
                        # function_value is  1 - acc on validation set (see hpobench.benchmarks.ml.README.md)
                        # output = output_dict["function_value"]  # TODO which metric to use?

                        output = 1 - output_dict["info"][seed]["test_scores"][cfg.global_min_metric]
                        outs.append(output)
                    output = np.mean(outs)
                    return output
            else:
                def target_function(configuration: Configuration, seed: int | None = None, **kwargs) -> float:
                    seed = rng.choice(seeds)
                    # printr(seed)
                    output_dict = benchmark.objective_function(configuration=configuration, seed=seed, **kwargs)
                    # function_value is  1 - acc on validation set (see hpobench.benchmarks.ml.README.md)
                    # output = output_dict["function_value"]  # TODO which metric to use?

                    output = 1 - output_dict["info"][seed]["test_scores"][cfg.global_min_metric]
                    # printr(output)
                    return output
            

            configuration_space = benchmark.get_configuration_space(seed=cfg.seed)

            x_optimum = None
            y_optimum = benchmark.get_global_min(metric=cfg.global_min_of)[cfg.global_min_metric]  # this is already 1 - min

            instance = TargetInstance(
                target_function=target_function,
                configuration_space=configuration_space,
                x_optimum=x_optimum,
                y_optimum=y_optimum,
            )
            printr(instance.y_optimum)
            instance_set[i] = instance
    else:
        raise NotImplementedError(f"We only know the BBOB benchmark so far. Got '{cfg.benchmark}'.")

    return instance_set
