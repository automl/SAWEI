from omegaconf import OmegaConf

from awei.baseline import make_schedule
from awei.baselines.portfolio_allocation import instantiate_acq_funs


OmegaConf.register_new_resolver("make_schedule", make_schedule)
OmegaConf.register_new_resolver("instantiate_acq_funs", instantiate_acq_funs)
