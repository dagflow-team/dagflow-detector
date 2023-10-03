from math import sqrt
from typing import TYPE_CHECKING

from numba import float64, njit, void
from numpy import double
from numpy.typing import NDArray

from dagflow.nodes import FunctionNode

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


@njit(
    void(
        float64,
        float64,
        float64,
        float64[:],
        float64[:],
    ),
    cache=True,
)
def _RelSigma(
    a: double,
    b: double,
    c: double,
    Energy: NDArray[double],
    Sigma: NDArray[double],
):
    a2 = a**2
    b2 = b**2
    for i in range(len(Energy)):
        Sigma[i] = sqrt(
            a2 + b2 / Energy[i] + (c / Energy[i]) ** 2
        )  # sqrt(a^2 + b^2/E + c^2/E^2)


class EnergyResolutionSigmaRelABC(FunctionNode):
    r"""
    Energy resolution $\sqrt(a^2 + b^2/E + c^2/E^2)$

    inputs:
        `a_nonuniform`: parameter a, due to energy deposition nonuniformity (size=1)
        `b_stat`: parameter b, due to stat fluctuations (size=1)
        `c_noise`: parameter c, due to dark noise (size=1)
        `0` or `Energy`: Input bin Energy (N elements)

    outputs:
        `0` or `RelSigma`: relative RelSigma for each bin (N elements)
    """

    __slots__ = ("_a_nonuniform", "_b_stat", "_c_noise", "_Energy", "_RelSigma")

    _a_nonuniform: "Input"
    _b_stat: "Input"
    _c_noise: "Input"
    _Energy: "Input"
    _RelSigma: "Output"

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Relative energy resolution σ/E",
                "latex": r"Relative energy resolution $\sigma/E$",
                "axis": r"$\sigma/E$",
            }
        )
        self._a_nonuniform, self._b_stat, self._c_noise = self._add_inputs( # pyright: ignore reportGeneralTypeIssues
            ("a_nonuniform", "b_stat", "c_noise"), positional=False
            )
        self._Energy = self._add_input("Energy")  # input: 0
        self._RelSigma = self._add_output("RelSigma")  # output: 0

    def _fcn(self) -> None:
        _RelSigma(
            self._a_nonuniform.data[0],
            self._b_stat.data[0],
            self._c_noise.data[0],
            self._Energy.data,
            self._RelSigma.data,
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import (
            AllPositionals,
            assign_output_axes_from_inputs,
            check_input_dimension,
            check_input_shape,
            copy_from_input_to_output,
        )

        check_input_shape(self, ("a_nonuniform", "b_stat", "c_noise"), (1,))
        check_input_dimension(self, AllPositionals, 1)
        copy_from_input_to_output(self, "Energy", "RelSigma")
        assign_output_axes_from_inputs(self, "Energy", "RelSigma", assign_meshes=True)
