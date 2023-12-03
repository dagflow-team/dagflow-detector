from numpy import isclose, linspace, matmul
from pytest import mark

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib import Array
from dagflow.plot import plot_array_1d, savefig#, showfig
from detector.RebinMatrix import RebinMatrix


# @mark.parametrize(
#    "dtype",
#    (
#        "d",
#        "f",
#    ),
# )
@mark.parametrize("m", (-5, -1, 1, 5))
def test_RebinMatrix(testname, m, dtype="d"):
    n = 11
    edges_old = linspace(-1.0, 1.0, n)
    edges_new = linspace(-1.0, 1.0, n+m)

    with Graph(close=True) as graph:
        EdgesOld = Array("EdgesOld", edges_old)
        EdgesNew = Array("EdgesNew", edges_new)
        mat = RebinMatrix("Rebin Matrix")
        EdgesOld >> mat("EdgesOld")
        EdgesNew >> mat("EdgesNew")

    res = mat.get_data()
    #ressum = res.sum(axis=0) # TODO: check matrix correctness

    y_old = linspace(0.,2., n-1)
    y_new = matmul(res, y_old)
    assert isclose(y_new.sum(), y_old.sum())
    
    plot_array_1d(array=y_old, edges=edges_old, yerr=0.5, color="blue")
    plot_array_1d(array=y_new, edges=edges_new, yerr=0.5, color="orange", linestyle="--")
    savefig(f"output/{testname}-plot.png")
    
    savegraph(graph, f"output/{testname}-graph.png")
