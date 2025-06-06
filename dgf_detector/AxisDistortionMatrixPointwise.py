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


@njit
def _project_y_to_x_linear(y: float, x0: float, x1: float, y0: float, y1: float):
    k = (y1 - y0) / (x1 - x0)
    if k == 0:
        return (x0 + x1) * 0.5  # TODO: check

    return (y - y0) / k + x0


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
    n_bins_x = edges_original.size - 1
    n_bins_y = edges_target.size - 1

    matrix[:, :] = 0.0

    n_points = distortion_original.size
    idx_last_point = n_points - 1
    last_x = distortion_original[idx_last_point]
    last_y = distortion_target[idx_last_point]

    large_positive_number = 1e100
    large_negative_number = -large_positive_number

    idx = -1
    x0, x1 = large_negative_number, large_negative_number
    y0, y1 = large_negative_number, large_negative_number

    bin_idx_x = 0
    bin_idx_y = 0
    left_x = edges_original[bin_idx_x]
    right_x = edges_original[bin_idx_x + 1]
    width_x_full = right_x - left_x

    left_y: float = edges_original[bin_idx_y]
    right_y: float = edges_original[bin_idx_y + 1]
    left, right = 0.0, 0.0
    # Find the starting point
    while idx < idx_last_point:
        if x1 > left_x:
            left = left_x
            break
        elif y1 > left_y:
            left = _project_y_to_x_linear(left_y, x0, x1, y0, y1)
            break

        idx += 1
        x0 = distortion_original[idx]
        y0 = distortion_target[idx]
        x1 = distortion_original[idx + 1]
        y1 = distortion_target[idx + 1]

        assert x1 > x0, "Allow only ascending x"

    # Advance:
    # - segments of the distortion curve: forward
    # - X bins: forward
    # - Y bins: forward/backward
    while idx < idx_last_point:
        passed_x = x1 > right_x
        passed_y_right = y1 > right_y
        passed_y_left = y1 < left_y

        assert not (passed_y_left & passed_y_right), "Can not pass left and right edge on Y at the same time"

        passed_any = False

        passed_x_first = False
        passed_y_right_first = False
        passed_y_left_first = False

        if passed_x:
            passed_any = True
            right = right_x
            passed_x_first = True

        if passed_y_right:
            right_x_from_y_right = _project_y_to_x_linear(right_y, x0, x1, y0, y1)
            if passed_any:
                if right_x_from_y_right==right:
                    passed_y_right_first = True
                elif right_x_from_y_right<right:
                    passed_x_first = False
                    passed_y_right_first = True
                    # passed_y_left_first = False

                    right = right_x_from_y_right
            else:
                passed_y_right_first = True
                right = right_x_from_y_right
                passed_any = True
        elif passed_y_left:
            right_x_from_y_left = _project_y_to_x_linear(left_y, x0, x1, y0, y1)
            if passed_any:
                if right_x_from_y_left==right:
                    passed_y_left_first = True
                elif right_x_from_y_left<right:
                    passed_x_first = False
                    # passed_y_right_first = False
                    passed_y_left_first = True

                    right = right_x_from_y_left
            else:
                passed_y_left_first = True
                right = right_x_from_y_left
                passed_any = True

        # Uncomment the following lines to see the debug output
        print(
                f"seg {idx:04d} x {x0:0.2g},{x1:0.2g} → y {y0:0.2g},{y1:0.2g}"
                " "
                f"ex {bin_idx_x:02d} {left_x:0.2g}→{right_x:0.2g}"
                " "
                f"ey {bin_idx_y:02d} {left_y:0.2g}→{right_y:0.2g}"
                " "
                f"p {passed_any:d} "
                f"X{passed_x:d}{passed_x_first:d} "
                f"Y{passed_y_right:d}{passed_y_right_first:d} "
                f"y{passed_y_left:d}{passed_y_left_first:d}"
                " "
                f"f {left:0.2g}→{right:0.2g}"
        )

        if passed_any:
            width_x_partial = fabs(right - left)
            matrix[bin_idx_y, bin_idx_x] = width_x_partial / width_x_full
            left = right

            if passed_x_first:
                bin_idx_x += 1
                if bin_idx_x>=n_bins_x-1:
                    break
                left_x = edges_original[bin_idx_x]
                if left_x > last_x:
                    break

                right_x = edges_original[bin_idx_x + 1]
                width_x_full = right_x - left_x

            if passed_y_right_first:
                if bin_idx_y==n_bins_y-1:
                    continue

                bin_idx_y+=1
                left_y = edges_original[bin_idx_y]
                if left_y > last_y:
                    break
                right_y = edges_original[bin_idx_y + 1]
            elif passed_y_left_first:
                if bin_idx_y==0:
                    continue

                bin_idx_y-=1
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

_axisdistortion_pointwise_numba: Callable[
    [NDArray, NDArray, NDArray, NDArray, NDArray], None
] = njit(cache=True)(_axisdistortion_pointwise_python)
