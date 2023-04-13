import pytest
from omegaconf import DictConfig

from dacbo.instances import create_instance_set


@pytest.mark.parametrize(
    "benchmark, instance_set",
    [
        # Instance set is a list of instances
        ("BBOB", [{"fid": 3, "dimension": 4, "instance": 1}, {"fid": 3, "dimension": 5, "instance": 1}]),
        # Instance set is a single instance
        ("BBOB", {"fid": 3, "dimension": 4, "instance": 1}),
    ],
)
def test_bbob_instance_set_creation(benchmark: str, instance_set: dict | list[dict]):
    cfg = DictConfig({"benchmark": benchmark, "instance_set": instance_set})
    instance_set = create_instance_set(cfg)


def test_unknown_benchmark_error():
    cfg = DictConfig({"benchmark": "unkown_benchmark", "instance_set": {}})
    with pytest.raises(NotImplementedError):
        instance_set = create_instance_set(cfg)
