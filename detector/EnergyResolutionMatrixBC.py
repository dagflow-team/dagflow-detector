from math import exp, sqrt
from typing import TYPE_CHECKING

from numba import float64, njit, void
from numpy import double, pi
from numpy.typing import NDArray

from dagflow.nodes import FunctionNode

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


@njit(float64(float64, float64, float64), cache=True)
def __resolution(Etrue: double, Erec: double, RelSigma: double) -> double:
    _invtwopisqrt = 1.0 / sqrt(2.0 * pi)
    sigma = Etrue * RelSigma
    reldiff = (Etrue - Erec) / sigma
    return exp(-0.5 * reldiff * reldiff) * _invtwopisqrt / sigma


@njit(void(float64[:], float64[:], float64[:, :], float64), cache=True)
def _resolution(
    RelSigma: NDArray[double],
    Edges: NDArray[double],
    Result: NDArray[double],
    minEvents: float,
) -> None:
    bincenter = lambda i: (Edges[i] + Edges[i + 1]) / 2
    nbins = len(RelSigma)
    for itrue in range(nbins):
        isRightEdge = False
        Etrue = bincenter(itrue)
        relsigma = RelSigma[itrue]
        for jrec in range(nbins):
            Erec = bincenter(jrec)
            dErec = Edges[jrec + 1] - Edges[jrec]
            rEvents = dErec * __resolution(Etrue, Erec, relsigma)
            if rEvents < minEvents:
                if isRightEdge:
                    break
                continue
            isRightEdge = True
            Result[jrec, itrue] = rEvents


class EnergyResolutionMatrixBC(FunctionNode):
    """
    Energy resolution

    inputs:
        `0` or `Edges`: Input bin Edges (N elements)
        `1` or `RelSigma`: Relative Sigma value for each bin (N elements)

    outputs:
        `0` or `SmearMatrix`: SmearMatrixing weights (NxN)
    """

    __slots__ = ("_Edges", "_RelSigma", "_SmearMatrix", "_minEvents")

    _Edges: "Input"
    _RelSigma: "Input"
    _SmearMatrix: "Output"
    _minEvents: float

    def __init__(self, name, minEvents: float = 1e-10, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Energy resolution $E_{res}$, MeV",
                "plottitle": r"Energy resolution $E_{res}$, MeV",
                "latex": r"$E_{res}$, MeV",
                "axis": r"$E_{res}$, MeV",
            }
        )
        self._minEvents = minEvents
        self._Edges = self._add_input("Edges")  # input: 0
        self._RelSigma = self._add_input("RelSigma")  # input: 1
        self._SmearMatrix = self._add_output("SmearMatrix")  # output: 0

    @property
    def minEvents(self) -> float:
        return self._minEvents

    def _fcn(self):
        _resolution(
            self._RelSigma.data,
            self._Edges.data,
            self._SmearMatrix.data,
            self.minEvents,
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import (
            check_input_dimension,
            find_max_size_of_inputs,
            check_input_size,
            AllPositionals,
        )

        check_input_dimension(self, AllPositionals, 1)
        size = find_max_size_of_inputs(self, "RelSigma")
        check_input_size(self, "Edges", exact=size + 1)

        RelSigmadd = self._RelSigma.dd
        self._SmearMatrix.dd.shape = (RelSigmadd.shape[0], RelSigmadd.shape[0])
        self._SmearMatrix.dd.dtype = RelSigmadd.dtype
        edges = self._Edges._parent_output
        self._SmearMatrix.dd.axes_edges = (edges, edges)
