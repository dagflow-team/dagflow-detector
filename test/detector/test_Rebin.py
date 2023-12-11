from numpy import allclose, finfo, linspace, matmul
from numpy.typing import NDArray
from pytest import mark, raises

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.plot import closefig, plot_array_1d, savefig
from detector.Rebin import Rebin
from detector.RebinMatrix import RebinMatrix


def partial_sum(y_old: NDArray, m: int) -> list:
    psum = []
    i = 0
    while i < y_old.size:
        psum.append(y_old[i : i + m].sum())
        i += m
    return psum


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("m", (2, 4))
@mark.parametrize("mode", ("python", "numba"))
def test_Rebin(testname, m, dtype, mode):
    n = 21
    edges_old = linspace(0.0, 2.0, n, dtype=dtype)
    edges_new = edges_old[::m]
    y_old = linspace(2.0, 0.0, n - 1, dtype=dtype)

    with Graph(close=True) as graph:
        EdgesOld = Array("edges_old", edges_old)
        EdgesNew = Array("edges_new", edges_new)
        Y = Array("Y", y_old)
        metanode = Rebin(mode=mode)
        # connect by metanode.inputs
        EdgesOld >> metanode.inputs["edges_old"]
        EdgesNew >> metanode.inputs["edges_new"]
        Y >> metanode.inputs["vector"]
        # or connect by certain node
        # edges_old >> metanode("edges_old", nodename="RebinMatrix")
        # edges_new >> metanode("edges_new", nodename="RebinMatrix")
        # Y >> metanode("vector", nodename="VectorMatrixProduct")

    mat = metanode.outputs["matrix"].data
    # NOTE: Asserts below are only for current edges_new! For other binning it may not coincide!
    assert (mat.sum(axis=0) == 1).all()
    assert mat.sum(axis=0).sum() == n - 1

    y_new = metanode.outputs["result"].data
    y_res = matmul(mat, y_old)
    assert all(y_res == y_new)

    rtol = finfo(dtype).resolution
    assert allclose(partial_sum(y_old, m), y_new, atol=0.0, rtol=rtol)

    # plots
    plot_array_1d(array=y_old, edges=edges_old, yerr=0.5, color="blue")
    plot_array_1d(array=y_new, edges=edges_new, yerr=0.5, color="orange", linestyle="--")
    savefig(f"output/{testname}-plot.png")
    closefig()

    savegraph(graph, f"output/{testname}-graph.png")


@mark.parametrize(
    "edges_new",
    (
        linspace(-1.0, 2.0, 21),
        linspace(0.0, 2.1, 21),
        linspace(0.0, 2.0, 41),
        linspace(0.0, 2.0, 10),
    ),
)
@mark.parametrize("mode", ("python", "numba"))
def test_RebinMatrix_wrong_edges_new(edges_new, mode):
    edges_old = linspace(0.0, 2.0, 21)
    with Graph(close=True):
        EdgesOld = Array("edges_old", edges_old)
        EdgesNew = Array("edges_new", edges_new)
        mat = RebinMatrix("Rebin Matrix", mode=mode)
        EdgesOld >> mat("edges_old")
        EdgesNew >> mat("edges_new")
    with raises(Exception):
        mat.get_data()
