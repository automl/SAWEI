
"""
Tabular benchmark
=================

This examples shows the usage of the containerized tabular benchmark.
To note: You don't have to pass the container name to the Benchmark-Constructor. It is automatically set, but for
demonstration purpose, we show how to set it.

container_source can be either a path to a registry (e.g. sylabs.io, singularity_hub.org) or a local path on your local
file system. If it is a link to a registry, the container will be downloaded to the default data dir, set in the
hpobenchrc. A second call, will first look into the data directory, if the container is already available, so it will not
be downloaded twice.

Please install the necessary dependencies via ``pip install .`` and singularity (v3.5).
https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
"""

import argparse
import os
os.environ["HPOBENCH_DEBUG"] = "true" 

from hpobench.container.benchmarks.ml.tabular_benchmark import TabularBenchmark as TabBenchmarkContainer
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
# from hpobench.container.benchmarks.nas.tabular_benchmarks import SliceLocalizationBenchmark as TabBenchmarkContainer
from hpobench.util.openml_data_manager import get_openmlcc18_taskids
from rich import inspect
from rich import print as printr

available_task_ids = {
    "rf"  : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168329, 168330, 168331, 168335, 168868, 168908, 168910, 168911, 168912],
    "xgb" : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168911, 168912],
    "svm" : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168329, 168330, 168331, 168335, 168868, 168908, 168909, 168910, 168911, 168912],
    "lr"  : [3, 12, 31, 53, 3917, 7592, 9952, 9977, 9981, 10101, 14965, 146195, 146212, 146606, 146818, 146821, 146822, 167119, 167120, 168329, 168330, 168331, 168335, 168868, 168908, 168909, 168910, 168911, 168912],
    "nn"  : [31, 53, 3917, 9952, 10101, 146818, 146821, 146822],
}


def run_experiment(on_travis=False):
    # task_ids = get_openmlcc18_taskids()
    # printr(task_ids)


    model = "nn"
    task_id = available_task_ids[model][0]

    printr(task_id)

    # benchmark = TabBenchmarkContainer(
    #     # container_name='tabular_benchmarks',
    #     # # benchmark_name="TabularBenchmark",
    #     model='rf',
    #     task_id = task_id,
    #     container_tag="benchmark",
    #     # container_source='library://phmueller/automl',              # gitlab.tf.uni-freiburg.de:5050/muelleph/hpobench-registry',
    #     container_name="ml_tabular", 
    #     container_source='/home/benjamin/Dokumente/code/tmp/HPOBench/hpobench/container/recipes/ml', # path to hpobench/container/recipes/ml
    #     rng=1,
    # )

    benchmark = TabularBenchmark(
        model=model, 
        task_id=task_id,
        rng=1
    )
    
    inspect(benchmark)

    cs = benchmark.get_configuration_space(seed=1)
    config = cs.sample_configuration()
    print(config)

    # You can pass the configuration either as a dictionary or a ConfigSpace.configuration
    result_dict_1 = benchmark.objective_function(configuration=config.get_dictionary())
    result_dict_2 = benchmark.objective_function(configuration=config)
    printr(result_dict_1)
    printr(result_dict_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='TabularNad')

    parser.add_argument('--on_travis', action='store_true',
                        help='Flag to speed up the run on the continuous integration tool \"travis\". This flag can be'
                             'ignored by the user')

    args = parser.parse_args()
    run_experiment(on_travis=args.on_travis)