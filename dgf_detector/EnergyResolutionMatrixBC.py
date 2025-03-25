from __future__ import annotations

from math import exp, sqrt
from typing import TYPE_CHECKING

from numba import njit
from numpy import allclose, pi

from dagflow.core.node import Node
from dagflow.core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_inputs_have_same_shape,
    check_size_of_inputs,
    find_max_size_of_inputs,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from dagflow.core.input import Input
    from dagflow.core.output import Output


@njit(cache=True)
def __resolution(Etrue: double, Erec: double, RelSigma: double) -> double:
    _invtwopisqrt = 1.0 / sqrt(2.0 * pi)
    sigma = Etrue * RelSigma
    reldiff = (Etrue - Erec) / sigma
    return exp(-0.5 * reldiff * reldiff) * _invtwopisqrt / sigma


@njit(cache=True)
def _resolution(
    RelSigma: NDArray[double],
    Edges: NDArray[double],
    EdgesOut: NDArray[double],
    Result: NDArray[double],
    minEvents: float,
) -> None:
    assert Edges is EdgesOut or allclose(Edges, EdgesOut, atol=0.0, rtol=0.0)

    bincenter = lambda i: (Edges[i] + Edges[i + 1]) * 0.5
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
                    Result[jrec:, itrue] = 0.0
                    break
                Result[jrec, itrue] = 0.0
                continue
            isRightEdge = True
            Result[jrec, itrue] = rEvents


class EnergyResolutionMatrixBC(Node):
    """Energy resolution.

    inputs:
        `0` or `RelSigma`: Relative Sigma value for each bin (N elements)
        `Edges`: Input bin Edges (N elements)
        `EdgesOut`: Output bin Edges (N elements), should be consistent with Edges.

    outputs:
        `0` or `SmearMatrix`: SmearMatrixing weights (NxN)
    """

    __slots__ = ("_Edges", "_EdgesOut", "_RelSigma", "_SmearMatrix", "_minEvents")

    _Edges: Input
    _EdgesOut: Input
    _RelSigma: Input
    _SmearMatrix: Output
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
        self._RelSigma = self._add_input("RelSigma")  # input: 0
        self._Edges = self._add_input("Edges", positional=False)
        self._EdgesOut = self._add_input("EdgesOut", positional=False)
        self._SmearMatrix = self._add_output("SmearMatrix")  # output: 0

    @property
    def minEvents(self) -> float:
        return self._minEvents

    def _function(self):
        _resolution(
            self._RelSigma.data,
            self._Edges.data,
            self._EdgesOut.data,
            self._SmearMatrix._data,
            self.minEvents,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dimension_of_inputs(self, AllPositionals, 1)
        size = find_max_size_of_inputs(self, "RelSigma")
        check_size_of_inputs(self, "Edges", exact=size + 1)
        check_inputs_have_same_shape(self, ["Edges", "EdgesOut"])

        RelSigmadd = self._RelSigma.dd
        self._SmearMatrix.dd.shape = (RelSigmadd.shape[0], RelSigmadd.shape[0])
        self._SmearMatrix.dd.dtype = RelSigmadd.dtype
        edges = self._Edges._parent_output
        edges_out = self._EdgesOut._parent_output
        self._SmearMatrix.dd.axes_edges = (edges_out, edges)
