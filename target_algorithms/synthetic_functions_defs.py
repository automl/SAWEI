from enum import Enum, auto

from abstract_synthetic_function import pybenchfcn_sop_class_constructor
from PyBenchFCN import SingleObjectiveProblem as SOP
from synthetic_functions import Branin


class SyntheticFunctions(Enum):
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


SF = SyntheticFunctions

synthetic_function_factory = {
    SF.ROSENBROCK: pybenchfcn_sop_class_constructor(sop_class=SOP.rosenbrockfcn),
    SF.ACKLEY: pybenchfcn_sop_class_constructor(sop_class=SOP.ackleyfcn),
    SF.EGGHOLDER: pybenchfcn_sop_class_constructor(sop_class=SOP.eggholderfcn),
    SF.BRANIN: Branin,
    # SF.SIXHUMPCAMEL: sixhumpcamelfcn),
    # SF.QUADRATIC: quadratic),
    # SF.HARTMANN3: hartmann3),
    SF.BUKINN6: pybenchfcn_sop_class_constructor(sop_class=SOP.bukinn6fcn),
    SF.CROSSINTRAY: pybenchfcn_sop_class_constructor(sop_class=SOP.crossintrayfcn),
    SF.DROPWAVE: pybenchfcn_sop_class_constructor(sop_class=SOP.dropwavefcn),
    SF.GRIEWANK: pybenchfcn_sop_class_constructor(sop_class=SOP.griewankfcn),
    SF.RASTRIGIN: pybenchfcn_sop_class_constructor(sop_class=SOP.rastriginfcn),
    SF.HOLDERTABLE: pybenchfcn_sop_class_constructor(sop_class=SOP.holdertablefcn),
    SF.SCHWEFEL: pybenchfcn_sop_class_constructor(sop_class=SOP.schwefelfcn),
    SF.LEVIN13: pybenchfcn_sop_class_constructor(sop_class=SOP.levin13fcn),
    SF.SCHAFFERN2: pybenchfcn_sop_class_constructor(sop_class=SOP.schaffern2fcn),
    SF.SCHAFFERN4: pybenchfcn_sop_class_constructor(sop_class=SOP.schaffern4fcn),
    SF.BOHACHEVSKY1: pybenchfcn_sop_class_constructor(sop_class=SOP.bohachevskyn1fcn),
    SF.BOHACHEVSKY2: pybenchfcn_sop_class_constructor(sop_class=SOP.bohachevskyn2fcn),
    # SF.BOHACHEVSKY3: bohachevskyn3fcn),
    SF.SPHERE: pybenchfcn_sop_class_constructor(sop_class=SOP.spherefcn),
}
