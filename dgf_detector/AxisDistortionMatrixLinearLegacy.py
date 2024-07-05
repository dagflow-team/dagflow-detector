from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.node import Node
from dagflow.typefunctions import (
    check_input_dimension,
    check_input_size,
    check_inputs_same_dtype,
    check_inputs_same_shape,
    copy_input_dtype_to_output,
    eval_output_dtype,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from dagflow.input import Input
    from dagflow.output import Output


class AxisDistortionMatrixLinearLegacy(Node):
    """For a given historam and distorted X axis compute the conversion matrix. Distortion is assumed to be linear.
    This is a legacy version of AxisDistortionMatrixLinear to be compatible with GNA implementation.
    """

    __slots__ = (
        "_edges_original",
        "_edges_modified",
        "_min_value_modified",
        "_result",
    )

    _edges_original: Input
    _edges_modified: Input
    _min_value_modified: float
    _result: Output

    def __init__(self, *args, min_value_modified: float = -1.0e10, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Bin edges distortion matrix",
            }
        )
        self._edges_original = self._add_input("EdgesOriginal", positional=False)
        self._edges_modified = self._add_input("EdgesModified", positional=False)
        self._result = self._add_output("matrix")  # output: 0
        self._min_value_modified = min_value_modified

        self._functions.update(
            {
                "python": self._fcn_python,
                "numba": self._fcn_numba,
            }
        )

    def _fcn_python(self):
        _axisdistortion_linear_python(
            self._edges_original.data,
            self._edges_modified.data,
            self._result.data,
            self._min_value_modified,
        )

    def _fcn_numba(self):
        _axisdistortion_linear_numba(
            self._edges_original.data,
            self._edges_modified.data,
            self._result.data,
            self._min_value_modified,
        )

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        names_edges = ("EdgesOriginal", "EdgesModified")
        check_input_dimension(self, names_edges, 1)
        check_inputs_same_dtype(self, names_edges)
        (nedges,) = check_inputs_same_shape(self, names_edges)
        check_input_size(self, "EdgesOriginal", min=1)
        copy_input_dtype_to_output(self, "EdgesOriginal", "matrix")
        eval_output_dtype(self, names_edges, "matrix")

        self._result.dd.shape = (nedges - 1, nedges - 1)
        edges = self._edges_original.parent_output
        self._result.dd.axes_edges = (edges, edges)
        self.fcn = self._functions["numba"]


def _axisdistortion_linear_python(
    edges_original: NDArray, edges_modified: NDArray, matrix: NDArray, min_value_modified: float
):
    # in general, target edges may be different (fine than original), the code should handle it.
    edges_target = edges_original
    # min_original = edges_original[0]
    min_target = edges_target[0]
    nbinsx = edges_original.size - 1
    nbinsy = edges_target.size - 1

    min_target = max(min_value_modified, min_target)

    matrix[:, :] = 0.0

    threshold = -1e10
    # left_axis = 0
    right_axis = 0
    idxy0, idxy1, idxy = -1, -1, 0
    # leftx_fine = threshold
    lefty_fine = threshold
    while idxy0 < 0 or lefty_fine <= threshold or lefty_fine < min_target:
        left_edge_from_x = edges_modified[idxy0 + 1] < edges_target[idxy1 + 1]
        if left_edge_from_x:
            # leftx_fine = edges_original[idxy0 + 1]
            lefty_fine = edges_modified[idxy0 + 1]
            # left_axis = 0
            if (idxy0 := idxy0 + 1) >= nbinsx:
                return
        else:
            # leftx_fine = -1
            lefty_fine = edges_target[idxy1 + 1]
            # left_axis = 1
            if (idxy1 := idxy1 + 1) >= nbinsy:
                return

    width_coarse = edges_modified[idxy0 + 1] - edges_modified[idxy0]
    while True:
        right_modified = edges_modified[idxy0 + 1]
        right_target = edges_target[idxy1 + 1]

        if right_modified < right_target:
            righty_fine = right_modified
            # rightx_fine = edges_original[idxy0 + 1]
            right_axis = 0
        else:
            righty_fine = right_target
            # rightx_fine = -1
            right_axis = 1

        while lefty_fine >= edges_target[idxy + 1]:
            if (idxy := idxy + 1) > nbinsy:
                break

        #
        # Uncomment the following lines to see the debug output
        # (you need to also uncomment all the `left_axis` lines)
        #
        # width_fine = righty_fine-lefty_fine
        # factor = width_fine/width_coarse
        # print(
        #         f"x:{leftx_fine:8.4f}→{rightx_fine:8.4f} "
        #         f"ax:{left_axis}→{right_axis} idxy:{idxy0: 4d},{idxy1: 4d} idxy: {idxy: 4d} "
        #         f"y:{lefty_fine:8.4f}→{righty_fine:8.4f}/{edges_modified[idxy0]:8.4f}→{edges_modified[idxy0+1]:8.4f}="
        #         f"{width_fine:8.4f}/{width_coarse:8.4f}={factor:8.4g} "
        # )

        matrix[idxy, idxy0] = (righty_fine - lefty_fine) / width_coarse

        if right_axis == 0:
            if (idxy0 := idxy0 + 1) >= nbinsx:
                break
            # WARNING: the following condition skips the procession of the last column, which is partial
            if edges_modified[idxy0 + 1] > edges_target[-1]:
                break
            width_coarse = edges_modified[idxy0 + 1] - edges_modified[idxy0]
        elif (idxy1 := idxy1 + 1) >= nbinsx:
            break
        lefty_fine = righty_fine
        # leftx_fine = rightx_fine
        # left_axis = right_axis


from numba import njit

_axisdistortion_linear_numba: Callable[[NDArray, NDArray, NDArray, float], None] = njit(cache=True)(
    _axisdistortion_linear_python
)
