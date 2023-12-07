from numpy import finfo, isclose, linspace, matmul
from pytest import mark, raises

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.plot import closefig, plot_array_1d, savefig

from detector.Rebin import Rebin
from detector.RebinMatrix import RebinMatrix


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("m", (2, 4))
@mark.parametrize("mode", ("python", "numba"))
def test_Rebin(testname, m, dtype, mode):
    n = 21
    edges_old = linspace(0.0, 2.0, n, dtype=dtype)
    edges_new = edges_old[::m]
    y_old = linspace(2.0, 0.0, n - 1, dtype=dtype)

    with Graph(close=True) as graph:
        EdgesOld = Array("EdgesOld", edges_old)
        EdgesNew = Array("EdgesNew", edges_new)
        Y = Array("Y", y_old)
        metanode = Rebin(mode=mode)
        # connect by metanode.inputs
        EdgesOld >> metanode.inputs["EdgesOld"]
        EdgesNew >> metanode.inputs["EdgesNew"]
        Y >> metanode.inputs["vector"]
        # or connect by certain node
        # EdgesOld >> metanode("EdgesOld", nodename="RebinMatrix")
        # EdgesNew >> metanode("EdgesNew", nodename="RebinMatrix")
        # Y >> metanode("vector", nodename="VectorMatrixProduct")

    mat = metanode.outputs["Matrix"].data
    # NOTE: Asserts below are only for current edges_new! For other binning it may not coincide!
    assert (mat.sum(axis=0) == 1).all()
    assert mat.sum(axis=0).sum() == n - 1

    y_new = metanode.outputs["result"].data
    y_res = matmul(mat, y_old)
    assert all(y_res == y_new)
    # NOTE: only for current edges_new! for other binning it may not coincide!
    rtol = finfo(dtype).resolution
    assert isclose(y_old.sum(), y_new.sum(), atol=0, rtol=rtol)

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
def test_RebinMatrix_wrong_edges_new(edges_new):
    edges_old = linspace(0.0, 2.0, 21)
    with Graph(close=True):
        EdgesOld = Array("EdgesOld", edges_old)
        EdgesNew = Array("EdgesNew", edges_new)
        mat = RebinMatrix("Rebin Matrix")
        EdgesOld >> mat("EdgesOld")
        EdgesNew >> mat("EdgesNew")
    with raises(Exception):
        mat.get_data()
