from numpy import linspace, matmul  # , isclose
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.plot import plot_array_1d, savefig, closefig
from detector.RebinMatrix import RebinMatrix

# TODO: integrals and sums are not the same berfor and after a rebinning;
#      new bins do not contain last bins of the old ones


@mark.parametrize(
    "dtype",
    (
        "d",
        "f",
    ),
)
@mark.parametrize("m", (2, 4))
def test_RebinMatrix(testname, m, dtype):
    n = 20
    edges_old = linspace(0.0, 2.0, n, dtype=dtype)
    edges_new = edges_old[::m]
    # edges_new = [*edges_new, edges_new[-1]+edges_new[1]-edges_new[0]]

    with Graph(close=True) as graph:
        EdgesOld = Array("EdgesOld", edges_old)
        EdgesNew = Array("EdgesNew", edges_new)
        mat = RebinMatrix("Rebin Matrix")
        EdgesOld >> mat("EdgesOld")
        EdgesNew >> mat("EdgesNew")

    res = mat.get_data()
    # ressum = res.sum(axis=0)  # TODO: check matrix correctness
    # print(ressum)

    y_old = linspace(0.0, 2.0, n - 1)
    y_new = matmul(res, y_old)
    # int_new = y_new.sum()
    # int_old = y_old.sum()
    # assert isclose(int_old, int_new) # TODO: do we need a check of integrals or sums?

    plot_array_1d(array=y_old, edges=edges_old, yerr=0.5, color="blue")
    plot_array_1d(array=y_new, edges=edges_new, yerr=0.5, color="orange", linestyle="--")
    savefig(f"output/{testname}-plot.png")
    closefig()

    savegraph(graph, f"output/{testname}-graph.png")
