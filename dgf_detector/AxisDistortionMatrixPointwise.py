from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import allclose, fabs

from dagflow.core.node import Node
from dagflow.core.type_functions import (
    check_dimension_of_inputs,
    check_inputs_have_same_dtype,
    check_inputs_have_same_shape,
    check_size_of_inputs,
    copy_dtype_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from dagflow.core.input import Input
    from dagflow.core.output import Output


class AxisDistortionMatrixPointwise(Node):
    """For a given historam and distorted X axis compute the conversion matrix.

    Distortion is assumed to be linear.
    """

    __slots__ = (
        "_edges_original",
        "_edges_target",
        "_distortion_original",
        "_distortion_target",
        "_result",
    )

    _edges_original: Input
    _edges_target: Input
    _edges_modified: Input
    _result: Output

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Bin edges distortion matrix",
            }
        )
        self._edges_original = self._add_input("EdgesOriginal", positional=False)
        self._edges_target = self._add_input("EdgesTarget", positional=False)
        self._distortion_original = self._add_input(
            "DistortionOriginal", positional=False
        )  # X
        self._distortion_target = self._add_input(
            "DistortionTarget", positional=False
        )  # Y
        self._result = self._add_output("matrix")  # output: 0

        self._functions_dict.update(
            {
                "python": self._function_python,
                "numba": self._function_numba,
            }
        )

    def _function_python(self):
        _axisdistortion_pointwise_python(
            self._edges_original.data,
            self._edges_target.data,
            self._distortion_original.data,
            self._distortion_target.data,
            self._result._data,
        )

    def _function_numba(self):
        _axisdistortion_pointwise_numba(
            self._edges_original.data,
            self._edges_target.data,
            self._distortion_original.data,
            self._distortion_target.data,
            self._result._data,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        names_edges = (
            "EdgesOriginal",
            "EdgesTarget",
            "DistortionOriginal",
            "DistortionTarget",
        )
        check_dimension_of_inputs(self, names_edges, 1)
        check_inputs_have_same_dtype(self, names_edges)
        (nedges,) = check_inputs_have_same_shape(self, names_edges[:2])
        check_inputs_have_same_shape(self, names_edges[2:])
        check_size_of_inputs(self, "EdgesOriginal", min=2)
        check_size_of_inputs(self, "DistortionOriginal", min=2)
        copy_dtype_from_inputs_to_outputs(self, "EdgesOriginal", "matrix")
        evaluate_dtype_of_outputs(self, names_edges, "matrix")

        self._result.dd.shape = (nedges - 1, nedges - 1)
        edges_original = self._edges_original.parent_output
        edges_target = self._edges_target.parent_output
        self._result.dd.axes_edges = (edges_target, edges_original)
        self.function = self._functions_dict["numba"]
        self.function = self._functions_dict["python"]


def _axisdistortion_pointwise_python(
    edges_original: NDArray,
    edges_target: NDArray,
    distortion_original: NDArray,
    distortion_target: NDArray,
    matrix: NDArray,
):
    # in general, target edges may be different (finer than original), the code should be able to handle it.
    # but currently we just check that edges are the same.
    assert edges_original is edges_target or allclose(
        edges_original, edges_target, atol=0.0, rtol=0.0
    )
    min_target = edges_target[0]
    nbinsx = edges_original.size - 1
    nbinsy = edges_target.size - 1

    matrix[:, :] = 0.0

    npoints = distortion_original.size
    lastpoint = npoints - 1
    last_x = distortion_original[lastpoint]
    last_y = distortion_target[lastpoint]

    idx = -1
    x0, x1 = -1e100, -1e100
    y0, y1 = -1e100, -1e100

    bin_idx_x = 0
    bin_idx_y = 0
    left_x = edges_original[bin_idx_x]
    right_x = edges_original[bin_idx_x + 1]
    width_x_full = right_x - left_x

    left_y = edges_original[bin_idx_y]
    right_y = edges_original[bin_idx_y + 1]
    left, right = 0.0, 0.0
    # Find the starting point
    while idx < lastpoint:
        if x1 > left_x:
            left = left_x
            break
        elif y1 > left_y:
            k = (y1 - y0) / (x1 - x0)
            if k == 0:
                left = (x0 + x1) * 0.5  # TODO: check
            else:
                left = (left_y - y0) / k + x0  # TODO: check
            break

        idx += 1
        x0 = distortion_original[idx]
        y0 = distortion_target[idx]
        x1 = distortion_original[idx + 1]
        y1 = distortion_target[idx + 1]

        assert x1 > x0, "Allow only ascending x"

    while idx < lastpoint:
        if x1 > right_x:
            right_axis = 0  # x
            right = right_x

            width_x_partial = fabs(right - left)
            matrix[bin_idx_y, bin_idx_x] = width_x_partial / width_x_full

            bin_idx_x += 1
            left_x = edges_original[bin_idx_x]
            if left_x > last_x:
                break

            right_x = edges_original[bin_idx_x + 1]
            width_x_full = right_x - left_x

            continue
        elif y1 > right_y:
            right_axis = 1  # y

            k = (y1 - y0) / (x1 - x0)
            if k == 0:
                right = (x0 + x1) * 0.5  # TODO: check
            else:
                right = (right_y - y0) / k + x0  # TODO: check

            width_x_partial = fabs(right - left)
            matrix[bin_idx_y, bin_idx_x] = width_x_partial / width_x_full

            left_y = edges_original[bin_idx_y]
            if left_y > last_y:
                break

            right_y = edges_original[bin_idx_y + 1]

            continue

        idx += 1
        x0 = distortion_original[idx]
        y0 = distortion_target[idx]
        x1 = distortion_original[idx + 1]
        y1 = distortion_target[idx + 1]
        assert x1 > x0, "Allow only ascending x"

        # Uncomment the following lines to see the debug output
        # (you need to also uncomment all the `left_axis` lines)
        #
        # width_fine = righty_fine-lefty_fine
        # factor = width_fine/width_coarse
        # print(
        #         f"x:{leftx_fine:8.4f}→{rightx_fine:8.4f} "
        #         f"ax:{left_axis}→{right_axis} bin_idx_y:{bin_idx_y0: 4d},{bin_idx_y1: 4d} bin_idx_y: {bin_idx_y: 4d} "
        #         f"y:{lefty_fine:8.4f}→{righty_fine:8.4f}/{edges_modified[bin_idx_y0]:8.4f}→{edges_modified[bin_idx_y0+1]:8.4f}="
        #         f"{width_fine:8.4f}/{width_coarse:8.4f}={factor:8.4g} "
        # )

_axisdistortion_pointwise_numba: Callable[
    [NDArray, NDArray, NDArray, NDArray, NDArray], None
] = njit(cache=True)(_axisdistortion_pointwise_python)
