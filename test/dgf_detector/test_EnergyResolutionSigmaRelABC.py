#!/usr/bin/env python

from numpy import allclose, finfo, linspace

from dagflow.graph import Graph
from dagflow.graphviz import savegraph
from dagflow.lib.Array import Array
from dagflow.plot import plot_auto
from dgf_detector.EnergyResolutionSigmaRelABC import EnergyResolutionSigmaRelABC


def test_EnergyResolutionSigmaRelABC_v01(debug_graph, testname):
    weights = [0.016, 0.081, 0.026]
    Energy = linspace(1.0, 8.0, 200)
    parnames = ("a_nonuniform", "b_stat", "c_noise")
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        a, b, c = tuple(
            Array(name, [val], mark=name) for name, val in zip(parnames, weights)
        )
        energy = Array("E", Energy, mark="Energy")
        sigma = EnergyResolutionSigmaRelABC("EnergyResolutionSigmaRelABC")
        # binding
        for name, inp in zip(parnames, (a, b, c)):
            inp >> sigma(name)
        energy >> sigma

    res = sigma._RelSigma.data

    show = False
    close = not show
    plot_auto(sigma, show=show, close=close, save=f"output/{testname}_plot.png")

    cmpto = (
        weights[0] ** 2 + weights[1] ** 2 / Energy + (weights[2] / Energy) ** 2
    ) ** 0.5
    assert allclose(res, cmpto, rtol=0, atol=finfo("d").resolution)

    savegraph(graph, f"output/{testname}.png")
