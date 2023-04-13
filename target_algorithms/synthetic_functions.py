from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from target_algorithms.abstract_synthetic_function import (
    AbstractSyntheticFunction, construct_configuration_space)


class SumOfDifferentPowers(AbstractSyntheticFunction):
    name = "Sum of Different Powers"
    n_dim_max = None
    global_minimum = 0.0
    x_optimal = None

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        d = n_dim
        self.n_dim = n_dim
        lower_bounds = -1
        upper_bounds = 1
        defaults = None
        sample_logs = [False] * d
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        self.x_optimal = [0.0] * d

        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
        """
        Perm

        From https://www.sfu.ca/~ssurjano/rothyp.html

        Parameters
        ----------
        x : [float, float]
            Input.

        Returns
        -------
        f(x).

        """
        x = np.array(x)
        I = np.arange(1, self.n_dim + 1)
        f = np.sum(np.power(x, I + 1))

        return f


class RotatedHyperEllipsoid(AbstractSyntheticFunction):
    name = "Rotated Hyper-Ellipsoid"
    n_dim_max = None
    global_minimum = 0.0
    x_optimal = None

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        d = n_dim
        self.n_dim = n_dim
        lower_bounds = -65.536
        upper_bounds = 65.536
        defaults = None
        sample_logs = [False] * d
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        self.x_optimal = [0.0] * d

        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
        """
        Perm

        From https://www.sfu.ca/~ssurjano/rothyp.html

        Parameters
        ----------
        x : [float, float]
            Input.

        Returns
        -------
        f(x).

        """
        x = np.array(x)
        J = np.arange(1, self.n_dim + 1)[::-1]
        f = np.sum(J * np.square(x))

        return f


class Perm(AbstractSyntheticFunction):
    name = "Perm"
    n_dim_max = None
    global_minimum = 0
    x_optimal = None

    def __init__(
        self, n_dim: int = 2, beta: float = 0.5, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs
    ):
        d = n_dim
        self.n_dim = n_dim
        lower_bounds = -d
        upper_bounds = d
        defaults = None
        sample_logs = [False] * d
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        self.x_optimal = [1 / i for i in np.arange(1, d + 1)]
        self.beta = beta

        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
        """
        Perm

        From https://www.sfu.ca/~ssurjano/perm0db.html

        Parameters
        ----------
        x : [float, float]
            Input.

        Returns
        -------
        f(x).

        """
        x = np.array(x)
        J = np.arange(1, self.n_dim + 1)
        f = np.sum([np.square(np.sum((J + self.beta) * (np.power(x, i) - 1 / np.power(J, i)))) for i in J])

        return f


class Trid(AbstractSyntheticFunction):
    name = "Trid"
    n_dim_max = None
    global_minimum = None
    x_optimal = None

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        d = n_dim
        lower_bounds = -(d**2)
        upper_bounds = d**2
        defaults = None
        sample_logs = [False] * d
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        self.x_optimal = [i * (d + 1 - i) for i in np.arange(1, d + 1)]
        self.global_minimum = -d * (d + 4) * (d - 1) / 6

        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
        """
        Trid

        From https://www.sfu.ca/~ssurjano/trid.html

        Parameters
        ----------
        x : [float, float]
            Input.

        Returns
        -------
        f(x).

        """
        x = np.array(x)
        f = np.sum(np.square(x - 1)) - np.sum(x[:-1] * x[1:])

        return f


class Quadratic(AbstractSyntheticFunction):
    name = "Quadratic"
    n_dim_max = None
    global_minimum = 0.0
    x_optimal = 0.0

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        lower_bounds = -100
        upper_bounds = 100
        defaults = None
        sample_logs = [False] * n_dim
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
        x_squared = np.square(x)
        x_summed = np.sum(x_squared)
        return x_summed.item()


class SixHumpCamel(AbstractSyntheticFunction):
    name = "Six-Hump Camel"
    n_dim_max = 2
    global_minimum = -1.0316
    x_optimal = [(0.0898, -0.7126), (-0.0898, 0.7126)]

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        if n_dim != 2:
            raise ValueError(f"SixHumpCamel only works on 2 dimensions. Got {n_dim}.")

        lower_bounds = -3
        upper_bounds = 3
        defaults = None
        sample_logs = [False, False]
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
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


class Hartmann3(AbstractSyntheticFunction):
    name = "Hartmann3"
    n_dim_max = 3
    global_minimum = -3.86278
    x_optimal = [0.114614, 0.555649, 0.852547]

    def __init__(self, n_dim: int = 3, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        if n_dim != 3:
            raise ValueError(f"Hartmann-3 only works on 3 dimensions. Got {n_dim}.")

        lower_bounds = 0.0
        upper_bounds = 1.0
        defaults = None
        sample_logs = [False, False, False]
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
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


class BohachevskyN3(AbstractSyntheticFunction):
    name = "Bohachevsky N.13"
    n_dim_max = 2
    global_minimum = 0.0
    x_optimal = [0.0, 0.0]

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        lower_bounds = -100
        upper_bounds = 100
        defaults = [-30, 50]
        sample_logs = [False, False]
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
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


class Branin(AbstractSyntheticFunction):
    name = "Branin"
    n_dim_max = 2
    global_minimum = 0.397887
    x_optimal = [[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]

    def __init__(self, n_dim: int = 2, configuration_space_file: Optional[Union[str, Path]] = None, **kwargs):
        lower_bounds = [-5.0, 0.0]
        upper_bounds = [10.0, 15.0]
        defaults = [0.0, 0.0]
        sample_logs = [False, False]
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
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


class Sinc5(AbstractSyntheticFunction):
    """
    Normalized sinc function with 5 optima (1 global, 4 local).

    Parameters
    ----------
    n_dim : int = 1
    configuration_space_file
    parameters : Optional[Tuple[float, float]]
        [x_offset, y_offset]
    """

    name = "Sinc5"
    n_dim_max = 1
    global_minimum = -1.0
    x_optimal = 0.0

    def __init__(
        self,
        n_dim: int = 1,
        configuration_space_file: Optional[Union[str, Path]] = None,
        parameters: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        lower_bounds = -5.0
        upper_bounds = 5.0
        defaults = 2.0
        sample_logs = [False]
        self.configuration_space = construct_configuration_space(
            n_dim=n_dim,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            defaults=defaults,
            sample_logs=sample_logs,
        )
        x_offset: Optional[float] = None
        y_offset: Optional[float] = None
        if parameters is not None:
            if len(parameters) != 2:
                raise ValueError("parameters = [x_offset, y_offset] must be of length 2." f"(Got {parameters}).")
            x_offset, y_offset = parameters
        self.x_offset = 0.0
        if x_offset is not None:
            if not lower_bounds < x_offset < upper_bounds:
                raise ValueError(
                    f"x_offset = {x_offset} does not satisfy " f"{lower_bounds} < x_offset < {upper_bounds}."
                )
            self.x_offset = x_offset
            self.x_optimal = x_offset
        self.y_offset = 0.0
        if y_offset is not None:
            self.y_offset = y_offset
            self.global_minimum = self.global_minimum + y_offset
        super().__init__(n_dim=n_dim, configuration_space_file=configuration_space_file)

    def function(self, x):
        return -np.sinc(x - self.x_offset) + self.y_offset


synthetic_function_classes = [
    Trid,
    Quadratic,
    SixHumpCamel,
    Hartmann3,
    BohachevskyN3,
    Branin,
    Perm,
    RotatedHyperEllipsoid,
    SumOfDifferentPowers,
    Sinc5,
]
