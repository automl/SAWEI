from omegaconf import OmegaConf

from awei.baseline import make_schedule


OmegaConf.register_new_resolver("make_schedule", make_schedule)
