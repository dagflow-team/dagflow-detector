from typing import TYPE_CHECKING

from dagflow.exception import InitializationError
from dagflow.nodes import FunctionNode
from numba import float64
from numba import int64
from numba import njit
from numba import void
from numpy import double
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dagflow.input import Input
    from dagflow.output import Output


@njit(void(float64[:], float64[:], float64[:], float64, int64), cache=True)
def _monotonize_with_x(
    x: NDArray[double],
    y: NDArray[double],
    result: NDArray[double],
    gradient: float,
    index: int,
) -> None:
    # forward loop
    i = index
    direction = 1 if y[i + 1] > y[i] else -1
    while i < len(y) - 1:
        direction_current = 1 if y[i + 1] > result[i] else -1
        if direction == direction_current:
            result[i + 1] = y[i + 1]
        else:
            result[i + 1] = result[i] + direction * gradient * (x[i + 1] - x[i])  # fmt:skip
        i += 1

    # backward loop
    if index == 0:
        return
    i = index + 2
    while i > 0:
        direction_current = 1 if result[i] > y[i - 1] else -1
        if direction == direction_current:
            result[i - 1] = y[i - 1]
        else:
            result[i - 1] = result[i] - direction * gradient * (x[i] - x[i - 1])  # fmt:skip
        i -= 1


@njit(void(float64[:], float64[:], float64, int64), cache=True)
def _monotonize_without_x(
    y: NDArray[double],
    result: NDArray[double],
    gradient: float,
    index: int,
) -> None:
    # forward loop
    i = index
    direction = 1 if y[i + 1] > y[i] else -1
    while i < len(y) - 1:
        direction_current = 1 if y[i + 1] > result[i] else -1
        if direction == direction_current:
            result[i + 1] = y[i + 1]
        else:
            result[i + 1] = result[i] + direction * gradient
        i += 1

    # backward loop
    if index == 0:
        return
    i = index + 2
    while i > 0:
        direction_current = 1 if result[i] > y[i - 1] else -1
        if direction == direction_current:
            result[i - 1] = y[i - 1]
        else:
            result[i - 1] = result[i] - direction * gradient
        i -= 1


class Monotonize(FunctionNode):
    r"""
    Monotonizes a function.

    inputs:
        `y`: f(x) array
        `x` (**optional**): arguments array

    outputs:
        `0` or `result`: the resulting array

    constructor arguments:
        `index_fraction`: fraction of array to monotonize (must be >=0 and <1)
        `gradient`: set gradient to monotonize (takes absolute value)
    """

    __slots__ = ("_y", "_result", "_index_fraction", "_gradient", "_index")

    _y: "Input"
    _result: "Output"
    _index_fraction: float
    _gradient: float
    _index: int

    def __init__(
        self,
        name,
        index_fraction: float = 0,
        gradient: float = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(name, *args, **kwargs, allowed_kw_inputs=("y", "x"))
        # TODO: set labels
        # self.labels.setdefaults(
        #    {
        #        "text": "",
        #        "plottitle": "",
        #        "latex": "",
        #        "axis": "",
        #    }
        # )
        if index_fraction < 0 or index_fraction >= 1:
            raise InitializationError(
                f"`index_fraction` must be 0 <= x < 1, but given {index_fraction}",
                node=self,
            )
        self._index_fraction = index_fraction
        self._gradient = abs(gradient)
        self._y = self._add_input("y", positional=False)  # input: "y"
        self._result = self._add_output("result")  # output: 0
        self._functions.update({"with_x": self._fcn_with_x, "without_x": self._fcn_without_x})

    @property
    def gradient(self) -> float:
        return self._gradient

    @property
    def index_fraction(self) -> float:
        return self._index_fraction

    @property
    def index(self) -> int:
        return self._index

    def _fcn_with_x(self) -> None:
        y = self.inputs["y"].data
        x = self.inputs["x"].data
        result = self.outputs["result"].data
        _monotonize_with_x(x, y, result, self.gradient, self.index)

    def _fcn_without_x(self) -> None:
        y = self.inputs["y"].data
        result = self.inputs["result"].data
        _monotonize_without_x(y, result, self.gradient, self.index)

    def _typefunc(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        from dagflow.typefunctions import (
            check_input_dimension,
            check_inputs_same_shape,
            copy_from_input_to_output,
        )

        isGivenX = self.inputs.get("x") is not None
        inputsToCheck = ("x", "y") if isGivenX else "y"

        check_input_dimension(self, inputsToCheck, 1)
        check_inputs_same_shape(self, inputsToCheck)
        copy_from_input_to_output(self, "y", "result")

        self._index = int((self.inputs["y"].dd.shape[0] - 1) * self.index_fraction)
        self.fcn = self._functions["with_x" if isGivenX else "without_x"]
