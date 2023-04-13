from typing import Optional

from dacbo.bo_configuration_space import AcquisitionFunctions
from dacbo.env import DACBOEnv
from dacbo.rollout import rollout
from dacbo.utils.experiment_setup import (
    get_target_algorithm,
    read_instance_set,
    select_instance,
)
from dacbo.utils.smac_interaction import get_acq_fun


def get_single_quadratic_2d_env(budget: int, seed: Optional[int] = None) -> DACBOEnv:
    instance_set_kwargs = {
        "family_id": "single",
        "ta_set_id": "QUADRATIC",
        "n_dim": 1,
    }
    instance_set, instance_set_path = read_instance_set(instance_set_kwargs)
    instance_id = 0
    acquisition_function_id = AcquisitionFunctions.EI
    instance = select_instance(instance_set, instance_id)
    target_algorithm = get_target_algorithm(instance)
    acquisition_function, acquisition_function_kwargs = get_acq_fun(acquisition_function_id=acquisition_function_id)
    env = DACBOEnv(instance_set=instance_set, cutoff=budget, seed=seed)
    return env


seed = 724
budget = 10
env = get_single_quadratic_2d_env(budget=budget, seed=seed)

policy = [1, 0, 1, 0, 0, 1, 1, 0, 0, 0]
data = rollout(env=env, policy=policy)
