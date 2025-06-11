from typing import Literal

from matplotlib import pyplot as plt
from numpy import (
    allclose,
    array,
    digitize,
    fabs,
    finfo,
    linspace,
    ma,
    poly1d,
    polyfit,
    rint,
)
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.lib.common import Array
from dagflow.plot.graphviz import savegraph
from dagflow.plot.plot import add_colorbar
from dgf_detector.AxisDistortionMatrix import AxisDistortionMatrix
from dgf_detector.AxisDistortionMatrixLinear import AxisDistortionMatrixLinear
from dgf_detector.AxisDistortionMatrixPointwise import AxisDistortionMatrixPointwise


@mark.parametrize("xoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("yoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("inverse", (False, True))
@mark.parametrize("nsegments", (10, 21, 4))
@mark.parametrize("dtype", ("d", "f"))
def test_AxisDistortionMatrixPointwise(
    dtype: str,
    nsegments: int,
    inverse: bool,
    xoffset: float | int,
    yoffset: float | int,
    testname: str,
):
    nbins = 10
    edges = linspace(0, nbins, nbins + 1, dtype=dtype)
    x_fine = linspace(edges[0], edges[-1], nsegments + 1, dtype=dtype) + xoffset
    y_fine = x_fine + yoffset
    if inverse:
        y_fine[:] = y_fine[::-1]

    print(x_fine)
    print(y_fine)

    with Graph(close_on_exit=True) as graph:
        Edges = Array("Edges", edges, mode="fill")
        EdgesModified = Array("Edges modified", edges, mode="fill")

        Distortion = Array("Distortion", x_fine, mode="fill")
        DistortionModified = Array("Distortion modified", y_fine, mode="fill")

        mat = AxisDistortionMatrixPointwise("LSNL matrix (pointwise)")

        Edges >> mat.inputs["EdgesOriginal"]
        Edges >> mat.inputs["EdgesTarget"]

        Distortion >> mat.inputs["DistortionOriginal"]
        DistortionModified >> mat.inputs["DistortionTarget"]

    res = mat.get_data()

    plt.figure()
    ax = plt.subplot(111, xlabel="x", ylabel="y", title="")
    ax.set_aspect("equal")
    # ax.grid()
    ax.minorticks_on()
    ax.vlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)
    ax.hlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)

    res_m = ma.array(res, mask=(res == 0))
    cmbl = ax.matshow(
        res_m, vmin=0, vmax=1, extent=[edges[0], edges[-1], edges[-1], edges[0]]
    )
    add_colorbar(cmbl)

    ax.plot(x_fine, y_fine, "+-", color="magenta")
    plt.savefig(f"output/{testname}-plot.pdf")

    ressum = res.sum(axis=0)
    print("Obtained matrix sum:\n", ressum)

    # plt.show()

    # TODO: test on contents
    assert (0.0 <= res).all() and (res <= 1.0).all()
    assert (0.0 <= ressum).all() and (ressum <= 1.0).all()

    # xfirst = x_fine[0]
    # xlast = x_fine[-1]

    # ixfirst = digitize(xfirst, edges, right=False)
    # ixlast = digitize(xlast, edges, right=True) - 1
    # if inverse:
    #     ixlast -= int(rint(xoffset))
    # else:
    #     ixlast -= int(rint(yoffset))
    # if ixlast > ixfirst:
    #     assert (ressum[ixfirst + 1 : ixlast] == 1.0).all()

    plt.close()


@mark.parametrize("xoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("yoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("inverse", (False, True))
@mark.parametrize(
    "poly_points",
    (
        [0, 5, 5, 5, 10],
        [0, 6, 5, 4, 10],
        [0, 8, 5, 2, 10],
        [0, 15, 5, -5, 10],
        [0, 0, 0, 0, 0],
    ),
)
# @mark.parametrize("nsegments", (10, 21, 4))
@mark.parametrize("nsegments", (21,))
# @mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("dtype", ("d",))
def test_AxisDistortionMatrixPointwise_pol3(
    dtype: str,
    nsegments: int,
    inverse: bool,
    xoffset: float | int,
    yoffset: float | int,
    poly_points: list[int | float],
    testname: str,
):
    nbins = 10
    edges = linspace(0, nbins, nbins + 1, dtype=dtype)

    poly_points = array(poly_points)
    poly_coeffs = polyfit([0, 3, 5, 7, 10], poly_points, deg=3)
    poly = poly1d(poly_coeffs)
    x_fine = linspace(edges[0], edges[-1], nsegments + 1, dtype=dtype)
    y_fine = poly(x_fine) + yoffset
    x_fine += xoffset
    if inverse:
        y_fine[:] = y_fine[::-1]

    # print(x_fine)
    # print(y_fine)

    with Graph(close_on_exit=True) as graph:
        Edges = Array("Edges", edges, mode="fill")
        EdgesModified = Array("Edges modified", edges, mode="fill")

        Distortion = Array("Distortion", x_fine, mode="fill")
        DistortionModified = Array("Distortion modified", y_fine, mode="fill")

        mat = AxisDistortionMatrixPointwise("LSNL matrix (pointwise)")

        Edges >> mat.inputs["EdgesOriginal"]
        Edges >> mat.inputs["EdgesTarget"]

        Distortion >> mat.inputs["DistortionOriginal"]
        DistortionModified >> mat.inputs["DistortionTarget"]

    res = mat.get_data()

    plt.figure()
    ax = plt.subplot(111, xlabel="x", ylabel="y", title="")
    ax.set_aspect("equal")
    # ax.grid()
    ax.minorticks_on()
    ax.vlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)
    ax.hlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)

    res_m = ma.array(res, mask=(res == 0))
    cmbl = ax.matshow(
        res_m, vmin=0, vmax=1, extent=[edges[0], edges[-1], edges[-1], edges[0]]
    )
    add_colorbar(cmbl)

    ax.plot(x_fine, y_fine, "+-", color="magenta")
    plt.savefig(f"output/{testname}-plot.pdf")

    ressum = res.sum(axis=0)
    print("Obtained matrix sum:\n", ressum)

    # plt.show()

    assert (0.0 <= res).all() and (res <= 1.0).all()
    assert (0.0 <= ressum).all() and (ressum <= 1.0).all()

    plt.close()
