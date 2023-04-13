from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple, Union

from PyBenchFCN import SingleObjectiveProblem as SOP

from target_algorithms.abstract_synthetic_function import (
    AbstractTargetAlgorithm, pybenchfcn_sop_class_constructor)
from target_algorithms.synthetic_functions import (BohachevskyN3, Branin,
                                                   Hartmann3, Perm, Quadratic,
                                                   RotatedHyperEllipsoid,
                                                   Sinc5, SixHumpCamel,
                                                   SumOfDifferentPowers, Trid)


class TargetAlgorithms(Enum):
    # Synthetic Functions
    ROSENBROCK = auto()
    ACKLEY = auto()
    EGGHOLDER = auto()
    BRANIN = auto()
    SIXHUMPCAMEL = auto()
    QUADRATIC = auto()
    HARTMANN3 = auto()
    BUKINN6 = auto()
    CROSSINTRAY = auto()
    DROPWAVE = auto()
    GRIEWANK = auto()
    RASTRIGIN = auto()
    HOLDERTABLE = auto()
    SCHWEFEL = auto()
    LEVIN13 = auto()
    SCHAFFERN2 = auto()
    SCHAFFERN4 = auto()
    BOHACHEVSKY1 = auto()
    BOHACHEVSKY2 = auto()
    BOHACHEVSKY3 = auto()
    SPHERE = auto()
    SUMSQUARE = auto()
    TRID = auto()
    PERM = auto()
    ROTATEDHYPERELLIPSOID = auto()
    SUMOFDIFFERENTPOWERS = auto()
    SINC5 = auto()


TA = TargetAlgorithms

target_algorithm_factory = {
    # Synthetic Functions
    TA.ROSENBROCK: pybenchfcn_sop_class_constructor(sop_class=SOP.rosenbrockfcn),
    TA.ACKLEY: pybenchfcn_sop_class_constructor(sop_class=SOP.ackleyfcn),
    TA.EGGHOLDER: pybenchfcn_sop_class_constructor(sop_class=SOP.eggholderfcn),
    TA.BRANIN: Branin,
    TA.SIXHUMPCAMEL: SixHumpCamel,
    TA.QUADRATIC: Quadratic,
    TA.HARTMANN3: Hartmann3,
    TA.BUKINN6: pybenchfcn_sop_class_constructor(sop_class=SOP.bukinn6fcn),
    TA.CROSSINTRAY: pybenchfcn_sop_class_constructor(sop_class=SOP.crossintrayfcn),
    TA.DROPWAVE: pybenchfcn_sop_class_constructor(sop_class=SOP.dropwavefcn),
    TA.GRIEWANK: pybenchfcn_sop_class_constructor(sop_class=SOP.griewankfcn),
    TA.RASTRIGIN: pybenchfcn_sop_class_constructor(sop_class=SOP.rastriginfcn),
    TA.HOLDERTABLE: pybenchfcn_sop_class_constructor(sop_class=SOP.holdertablefcn),
    TA.SCHWEFEL: pybenchfcn_sop_class_constructor(sop_class=SOP.schwefelfcn),
    TA.LEVIN13: pybenchfcn_sop_class_constructor(sop_class=SOP.levin13fcn),
    TA.SCHAFFERN2: pybenchfcn_sop_class_constructor(sop_class=SOP.schaffern2fcn),
    TA.SCHAFFERN4: pybenchfcn_sop_class_constructor(sop_class=SOP.schaffern4fcn),
    TA.BOHACHEVSKY1: pybenchfcn_sop_class_constructor(sop_class=SOP.bohachevskyn1fcn),
    TA.BOHACHEVSKY2: pybenchfcn_sop_class_constructor(sop_class=SOP.bohachevskyn2fcn),
    TA.BOHACHEVSKY3: BohachevskyN3,
    TA.SPHERE: pybenchfcn_sop_class_constructor(sop_class=SOP.spherefcn),
    TA.SUMSQUARE: pybenchfcn_sop_class_constructor(sop_class=SOP.sumsquaresfcn),
    TA.TRID: Trid,
    TA.PERM: Perm,
    TA.ROTATEDHYPERELLIPSOID: RotatedHyperEllipsoid,
    TA.SUMOFDIFFERENTPOWERS: SumOfDifferentPowers,
    TA.SINC5: Sinc5,
}


def instantiate_target_algorithm(
    target_algorithm: TargetAlgorithms,
    n_dim: int = 2,
    configuration_space_file: Optional[Union[str, Path]] = None,
    parameters: Optional[Tuple] = None,
    **kwargs,
) -> AbstractTargetAlgorithm:
    ta_cls = target_algorithm_factory[target_algorithm]
    if type(configuration_space_file) == str and len(configuration_space_file) == 0:
        configuration_space_file = None
    ta = ta_cls(n_dim=n_dim, configuration_space_file=configuration_space_file, parameters=parameters)
    return ta


def print_n_dim_max():
    for ta_class in target_algorithm_factory.values():
        print(f"{ta_class.name:30s}: n_dim_max = {ta_class.n_dim_max}")
