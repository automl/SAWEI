from omegaconf import OmegaConf


def calculate_n_trials(budget_doe: int, budget_sbo: int) -> int:
    return budget_doe + budget_sbo


OmegaConf.register_new_resolver("calculate_n_trials", calculate_n_trials)
