from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig

from awei.abstract_policy import AbstractPolicy

def make_policy(cfg: DictConfig) -> AbstractPolicy:
    policy = instantiate(cfg["policy"])
    return policy
