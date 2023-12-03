from typing import TYPE_CHECKING

from dagflow.nodes import FunctionNode
from numpy import isclose
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


class RebinMatrix(FunctionNode):
    """For a given `edges_old` and `edges_new` computes the conversion matrix"""

    __slots__ = (
        "_edges_old",
        "_edges_new",
        "_result",
    )

    _edges_old: "Input"
    _edges_new: "Input"
    _result: "Output"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels.setdefaults(
            {
                "text": "Bin edges conversion matrix",
            }
        )
        self._edges_old = self._add_input("EdgesOld", positional=False)
        self._edges_new = self._add_input("EdgesNew", positional=False)
        self._result = self._add_output("matrix")  # output: 0
        self._functions.update(
            {
                "python": self._fcn_python,
                "numba": self._fcn_numba,
            }
        )

    def _fcn_python(self):
        _calc_rebin_matrix_python(
            self._edges_old.data,
            self._edges_new.data,
            self._result.data,
        )

    def _fcn_numba(self):
        _calc_rebin_matrix_numba(
            self._edges_old.data,
            self._edges_new.data,
            self._result.data,
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import check_input_dimension

        check_input_dimension(self, ("EdgesOld", "EdgesNew"), 1)
        self._result.dd.shape = (self._edges_new.dd.size - 1, self._edges_old.dd.size - 1)
        self._result.dd.dtype = "d"
        self.fcn = self._functions["python"]


def _calc_rebin_matrix_python(edges_old: NDArray, edges_new: NDArray, rebin_matrix: NDArray) -> None:
    """
    For a column C of size N: Cnew = M C
    Cnew = [Mx1]
    M = [MxN]
    C = [Nx1]
    """
    assert edges_new[0] >= edges_old[0] or isclose(edges_new[0], edges_old[0])
    assert edges_new[-1] <= edges_old[-1] or isclose(edges_new[-1], edges_old[-1])

    inew = 0
    iold = 0
    nold = edges_old.size
    edge_old = edges_old[0]
    edge_new_prev = edges_new[0]

    stepper_old = enumerate(edges_old)
    iold, edge_old = next(stepper_old)
    for inew, edge_new in enumerate(edges_new[1:], 1):
        while edge_old < edge_new and not isclose(edge_new, edge_old):
            if edge_old >= edge_new_prev or isclose(edge_old, edge_new_prev):
                rebin_matrix[inew - 1, iold] = 1.0

            iold, edge_old = next(stepper_old)
            if iold >= nold:
                # with printoptions(threshold = 100000):
                print("Old:", edges_old.size, edges_old)
                print("New:", edges_new.size, edges_new)
                raise RuntimeError(f"Inconsistent edges (outer): {iold} {edge_old}, {inew} {edge_new}")

        # TODO: why this check is here? It always raises an exception!
        # if not isclose(edge_new, edge_old):
        #    # with printoptions(threshold = 100000):
        #    print("Old:", edges_old.size, edges_old)
        #    print("New:", edges_new.size, edges_new)
        #    raise RuntimeError(f"Inconsistent edges (inner): {iold} {edge_old}, {inew} {edge_new}")


from typing import Callable  # fmt:skip
from numba import njit

_calc_rebin_matrix_numba: Callable[[NDArray, NDArray, NDArray], None] = njit(cache=True)(
    _calc_rebin_matrix_python
)
