# -*- coding: utf-8 -*-
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from PyBenchFCN import Factory
from PyBenchFCN import SingleObjectiveProblem as SOP

from target_algorithms.registration import (TargetAlgorithms,
                                            target_algorithm_factory)


def wrap(fn):
    def wrapper(config):
        # pass configuration values not by name but by order
        cfg = config.get_dictionary()
        vals = np.asarray(list(cfg.values()))

        return fn(vals)

    return wrapper


def quadratic(x):
    x_squared = np.square(x)
    x_summed = np.sum(x_squared)
    return x_summed


def branin(x):
    # from smac/examples/branin.py
    x1 = x[0]
    x2 = x[1]
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    ret = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return ret


def sixhumpcamelfcn(x) -> float:
    """
    Six-Hump Camel Function (2D)

    :math:`f(\mathbf{x}) = ( 4 - 2.1 {x_1}^2 + {x_1}^4 / 3 ) {x_1}^2  +  x_1 x_2  + (-4 + 4 {x_2}^2) {x_2}^2`

    Usually evaluated in:
        :math:`x_1 \in [-3, 3], x_2 \in [-2, 2]`

    Global minimum:
        :math:`f(\mathbf{x^*}) = -1.0316  \;\mathrm{at}\, \mathbf{x^*} = (0.0898, -0.7126) \,\mathrm{and}\, (-0.0898, 0.7126)`

    From https://www.sfu.ca/~ssurjano/camel6.html

    Parameters
    ----------
    x : [float, float]
        Input.

    Returns
    -------
    f(x).

    """
    x1 = x[0]
    x2 = x[1]

    x12 = x1 * x1
    x22 = x2 * x2

    f = (4 - 2.1 * x12 + (x12 * x12) / 3) * x12 + x1 * x2 + (-4 + 4 * x22) * x22

    return f


def hartmann3(x) -> float:
    """
    Hartmann-3 function (3D)

    The function has 4 local minima.

    Usually evaluated on the hypercube xi ∈ (0, 1), for all i = 1, 2, 3.

    Global minimum:
        :math:`f(\mathbf{x^*}) = -3.86278  \;\mathrm{at}\, \mathbf{x^*} = (0.114614, 0.555649, 0.852547)`

    From https://www.sfu.ca/~ssurjano/hart3.html

    Parameters
    ----------
    x : [float, float, float]
        Input.

    Returns
    -------
    float
        Function value at x.

    """
    if len(x) != 3:
        raise ValueError("Hartmann-3 only takes input vectors of length 3. Passed a vector " f"of length {len(x)}.")

    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    P = 10 ** (-4) * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])

    i_max = len(alpha)
    j_max = len(x)

    f = 0
    for i in range(i_max):
        exponent = 0
        for j in range(j_max):
            exponent -= A[i, j] * (x[j] - P[i, j]) ** 2
        f -= alpha[i] * np.exp(exponent)

    return f


def bohachevskyn3fcn(x):
    """
    https://www.sfu.ca/~ssurjano/boha.html
    """
    x = np.array(x)
    n_var = len(x)
    max_var = 2
    if n_var != max_var:
        raise ValueError("The Bohachevsky N. 3 function is only defined on a 2D space.")

    X = x[None, :]
    x1 = X[:, 0]
    x2 = X[:, 1]

    scores = np.power(x1, 2) + 2 * np.power(x2, 2) - 0.3 * np.cos(3 * np.pi * x1 + 4 * np.pi * x2) + 0.3
    return scores.item()


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
    SF.ROSENBROCK: SOP.rosenbrockfcn,
    SF.ACKLEY: SOP.ackleyfcn,
    SF.EGGHOLDER: SOP.eggholderfcn,
    SF.BRANIN: branin,
    SF.SIXHUMPCAMEL: sixhumpcamelfcn,
    SF.QUADRATIC: quadratic,
    SF.HARTMANN3: hartmann3,
    SF.BUKINN6: SOP.bukinn6fcn,
    SF.CROSSINTRAY: SOP.crossintrayfcn,
    SF.DROPWAVE: SOP.dropwavefcn,
    SF.GRIEWANK: SOP.griewankfcn,
    SF.RASTRIGIN: SOP.rastriginfcn,
    SF.HOLDERTABLE: SOP.holdertablefcn,
    SF.SCHWEFEL: SOP.schwefelfcn,
    SF.LEVIN13: SOP.levin13fcn,
    SF.SCHAFFERN2: SOP.schaffern2fcn,
    SF.SCHAFFERN4: SOP.schaffern4fcn,
    SF.BOHACHEVSKY1: SOP.bohachevskyn1fcn,
    SF.BOHACHEVSKY2: SOP.bohachevskyn2fcn,
    SF.BOHACHEVSKY3: bohachevskyn3fcn,
    SF.SPHERE: SOP.spherefcn,
}


def get_global_minimum(name, dim):
    global_minima = {
        # (Function type, number of dimension): global minimum
        (SF.BRANIN, 2): 0.397887,
        (SF.QUADRATIC, None): 0.0,
        (SF.ROSENBROCK, None): 0,
        (SF.EGGHOLDER, None): -959.6407,
        (SF.HARTMANN3, 3): -3.86278,
        (SF.SIXHUMPCAMEL, 2): -1.0316,
        (SF.ACKLEY, None): 0,
        (SF.BUKINN6, 2): 0,
        (SF.CROSSINTRAY, 2): -2.06261,
        (SF.DROPWAVE, 2): -1,
        (SF.GRIEWANK, None): 0,
        (SF.RASTRIGIN, None): 0,
        (SF.HOLDERTABLE, 2): -19.2085,
        (SF.SCHWEFEL, None): 0,
        (SF.LEVIN13, 2): 0,
        (SF.SCHAFFERN2, 2): 0,
        (SF.SCHAFFERN4, 2): 0.2925786328424814,
        (SF.BOHACHEVSKY1, 2): 0,
        (SF.BOHACHEVSKY2, 2): 0,
        (SF.BOHACHEVSKY3, 2): 0,
        (SF.SPHERE, None): 0,
    }
    keys = list(global_minima.keys())
    nd_funcs = [k[0] for k in keys if k[1] is None]
    if name in nd_funcs:  # [SF.QUADRATIC, SF.ROSENBROCK, SF.EGGHOLDER, SF.ACKLEY, SF.GRIEWANK]:
        dim = None

    if type(name) == SyntheticFunctions:
        # old style
        global_minimum = global_minima[(name, dim)]
    else:
        # new style
        global_minimum = target_algorithm_factory[name](n_dim=dim).global_minimum
    return global_minimum


def get_optfunc(test_function: SyntheticFunctions, n_dim: int = 2, **kwargs):
    sf = synthetic_function_factory[test_function]
    if "PyBenchFCN" in sf.__module__:
        sop = sf(n_var=n_dim)
        fn = wrap(sop.f)
    else:
        fn = wrap(sf)
    optfunc = fn

    return optfunc


class PlotSOP(object):
    def __init__(
        self,
        problem_name,
        plot_type="surface",
        mode="show",
        savepath=None,
        cmap="viridis",
        plot_bound=None,
        n_part=100,
        n_level=10,
        fig=None,
        ax=None,
    ):
        self.problem_name = problem_name
        self.mode = mode
        self.savepath = savepath
        self.plot_type = plot_type
        self.plot_bound = plot_bound
        self.n_part = n_part
        self.n_level = n_level
        self.cmap = cmap

        self.fig = fig
        if self.fig is None:
            self.fig = plt.figure()

        self.ax = ax

        # TODO adjust self.plot_type based on projection of axis

    def __call__(self):
        self._do()
        return self.fig, self.ax

    def _do(self):
        if self.plot_type == "surface":
            self._plot_surfaceSOP(self._calc_plot_dataSOP())
        if self.plot_type == "contour":
            self._plot_contourSOP(self._calc_plot_dataSOP())

    def _calc_plot_dataSOP(self):
        problem = Factory.set_sop(self.problem_name, n_var=2)  # Set benchmark problem
        if problem.n_var != 2:
            print("This function does not support 2D space.")
            return None
        if self.plot_bound == None:
            xl, xu = problem.plot_bound
        else:
            xl, xu = self.plot_bound

        xl = np.zeros(2) + xl
        xu = np.zeros(2) + xu

        x1 = np.linspace(xl[0], xu[0], self.n_part)  # Generate pairs of decision value
        x2 = np.linspace(xl[1], xu[1], self.n_part)  # Generate pairs of decision value
        X1, X2 = np.meshgrid(x1, x2)
        X = np.c_[np.ravel(X1), np.ravel(X2)]
        F = problem.F(X).reshape(len(X1), len(X2))

        return X1, X2, F

    def _savefig(self, fig):
        if not self.savepath:
            self.savepath = f"./{self.problem_name}_{self.plot_type}.svg"

        fig.savefig(self.savepath)

    def _plot_surfaceSOP(self, INPUT):
        if INPUT == None:
            return
        X1, X2, F = INPUT
        if self.ax is None:
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.plot_surface(X1, X2, F, cmap=self.cmap)
        if self.mode == "show":
            pass
            # plt.show()
        if self.mode == "save":
            self._savefig(fig)

    def _plot_contourSOP(self, INPUT):
        if INPUT == None:
            return
        X1, X2, F = INPUT
        if self.ax is None:
            self.ax = self.fig.add_subplot(111)
        contour = self.ax.contour(X1, X2, F, levels=self.n_level, cmap=self.cmap)
        self.fig.colorbar(contour)
        if self.mode == "show":
            pass
            # plt.show()
        if self.mode == "save":
            self._savefig(fig)


def plot_contour2d(
    test_function: SyntheticFunctions,
    fig=None,
    ax=None,
    plot_bound=None,
    cmap="coolwarm",
):
    fn = synthetic_function_factory[SF.ROSENBROCK]

    fig, ax = PlotSOP(
        fn.__name__.replace("fcn", ""),
        plot_type="contour",
        cmap=cmap,
        ax=ax,
        fig=fig,
        plot_bound=plot_bound,
    )()

    return fig, ax


if __name__ == "__main__":
    fn = get_optfunc(SF.BOHACHEVSKY3)
