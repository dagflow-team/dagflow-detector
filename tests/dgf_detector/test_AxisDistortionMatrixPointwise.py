from typing import Literal

from numpy import allclose, array, finfo
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.lib.common import Array
from dagflow.plot.graphviz import savegraph
from dgf_detector.AxisDistortionMatrix import AxisDistortionMatrix
from dgf_detector.AxisDistortionMatrixLinear import AxisDistortionMatrixLinear
from dgf_detector.AxisDistortionMatrixPointwise import AxisDistortionMatrixPointwise


# @mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("dtype", ("d",))
# @mark.parametrize("nsegments", (10, 5, 100))
@mark.parametrize("nsegments", (10,))
# @mark.parametrize("xoffset", (0, -0.5, +0.5, -5, +5))
# @mark.parametrize("yoffset", (0, -0.5, +0.5, -5, +5))
@mark.parametrize("xoffset", (0, -0.5, +0.5))
@mark.parametrize("yoffset", (0, -0.5, +0.5))
def test_AxisDistortionMatrixPointwise(
    dtype: str, nsegments: int, xoffset: float | int, yoffset: float | int
):
    nbins = 10
    edges = linspace(0, nbins, nbins + 1, dtype=dtype)
    x_fine = linespace(edges[0], edges[-1], nsegments + 1, dtype=dtype) + xoffset
    y_fine = x_fine + yoffset

    # with Graph(close_on_exit=True) as graph:
    #     Edges = Array("Edges", edges, mode="fill")
    #     EdgesModified = Array("Edges modified", edges_modified, mode="fill")

    #     mat = AxisDistortionMatrixPointwise("LSNL matrix (pointwise)")

    #     Edges >> mat.inputs["EdgesOriginal"]
    #     Edges >> mat.inputs["EdgesTarget"]

    #     Edges >> mat.inputs["DistortionOriginal"]
    #     EdgesModified >> mat.inputs["DistortionTarget"]

    # res = mat.get_data()

    # ressum = res.sum(axis=0)
    # print("Obtained matrix:\n", res)
    # print("Obtained matrix sum:\n", ressum)

    # if mode == "pointwise":
    #     atol = finfo(dtype).resolution * 0.5
    # else:
    #     atol = 0 if dtype == "d" else finfo(dtype).resolution * 0.5
    # assert allclose(res, desired, atol=atol, rtol=0)

    # idxstart, idxend = 0, nbins
    # while idxstart < nbins and ressum[idxstart] < 1.0:
    #     idxstart += 1
    # while idxend > 0 and ressum[idxend - 1] < 1.0:
    #     idxend -= 1
    # assert allclose(ressum[idxstart:idxend], 1, rtol=0, atol=0)

    # out_edges = mat.outputs[0].dd.axes_edges
    # assert out_edges[0] is out_edges[1]
    # assert out_edges[0] is Edges.outputs[0]

    # savegraph(
    #     graph,
    #     f"output/test_AxisDistortionMatrix{mode.capitalize()}_{dtype}.png",
    # )
