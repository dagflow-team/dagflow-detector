from __future__ import annotations

from typing import TYPE_CHECKING

from dagflow.exception import ConnectionError
from dagflow.lib import BinCenter
from dagflow.metanode import MetaNode
from dagflow.storage import NodeStorage
from dgf_detector.EnergyResolutionMatrixBC import EnergyResolutionMatrixBC
from dgf_detector.EnergyResolutionSigmaRelABC import EnergyResolutionSigmaRelABC

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dagflow.node import Node
    from multikeydict.typing import KeyLike


class EnergyResolution(MetaNode):
    __slots__ = (
        "_EnergyResolutionMatrixBCList",
        "_EnergyResolutionSigmaRelABCList",
        "_BinCenterList",
    )

    _EnergyResolutionMatrixBCList: list[Node]
    _EnergyResolutionSigmaRelABCList: list[Node]
    _BinCenterList: list[Node]

    def __init__(self, *, bare: bool = False, labels: Mapping = {}):
        super().__init__()
        self._EnergyResolutionMatrixBCList = []
        self._EnergyResolutionSigmaRelABCList = []
        self._BinCenterList = []
        if bare:
            return

        self.add_EnergyResolutionSigmaRelABC(
            name="EnergyResolutionSigmaRelABC",
            label=labels.get("EnergyResolutionSigmaRelABC", {}),
        )
        self.add_BinCenter("BinCenter", labels.get("BinCenter", {}))
        self.add_EnergyResolutionMatrixBC("EnergyResolution", labels.get("EnergyResolution", {}))
        self._bind_outputs()

    def add_EnergyResolutionSigmaRelABC(
        self,
        name: str = "EnergyResolutionSigmaRelABC",
        label: Mapping = {},
    ) -> EnergyResolutionSigmaRelABC:
        _EnergyResolutionSigmaRelABC = EnergyResolutionSigmaRelABC(name=name, label=label)
        self._EnergyResolutionSigmaRelABCList.append(_EnergyResolutionSigmaRelABC)
        self._add_node(
            _EnergyResolutionSigmaRelABC,
            kw_inputs=["Energy", "a_nonuniform", "b_stat", "c_noise"],
            kw_outputs=["RelSigma"],
            merge_inputs=["Energy"],
        )
        return _EnergyResolutionSigmaRelABC

    def add_BinCenter(self, name: str = "BinCenter", label: Mapping = {}) -> BinCenter:
        _BinCenter = BinCenter(name, label=label)
        _BinCenter._add_pair("Edges", "Energy")
        self._BinCenterList.append(_BinCenter)
        self._add_node(
            _BinCenter,
            kw_inputs=["Edges"],
            kw_outputs=["Energy"],
            merge_inputs=["Edges"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return _BinCenter

    def add_EnergyResolutionMatrixBC(
        self,
        name: str = "EnergyResolution",
        label: Mapping = {},
    ) -> EnergyResolutionMatrixBC:
        _EnergyResolutionMatrixBC = EnergyResolutionMatrixBC(name, label=label)
        self._EnergyResolutionMatrixBCList.append(_EnergyResolutionMatrixBC)
        self._add_node(
            _EnergyResolutionMatrixBC,
            kw_inputs=["RelSigma", "Edges"],
            kw_outputs=["SmearMatrix"],
            merge_inputs=["Edges"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return _EnergyResolutionMatrixBC

    def _bind_outputs(self) -> None:
        if not (
            (l1 := len(self._BinCenterList))
            == (l2 := len(self._EnergyResolutionMatrixBCList))
            == (l3 := len(self._EnergyResolutionSigmaRelABCList))
        ):
            raise ConnectionError(
                f"Cannot bind outputs! Nodes must be triplets of (BinCenter, EnergyResolutionMatrixBC, EnergyResolutionSigmaRelABC), but current lengths are {l1}, {l2}, {l3}!",
                node=self,
            )
        for _BinCenter, _EnergyResolutionSigmaRelABC, _EnergyResolutionMatrixBC in zip(
            self._BinCenterList,
            self._EnergyResolutionSigmaRelABCList,
            self._EnergyResolutionMatrixBCList,
        ):
            _BinCenter.outputs["Energy"] >> _EnergyResolutionSigmaRelABC.inputs["Energy"]
            _EnergyResolutionSigmaRelABC._RelSigma >> _EnergyResolutionMatrixBC.inputs["RelSigma"]

    # TODO: check this again; what should be in replicate_outputs argument: all the nodes or only main?
    @classmethod
    def replicate(
        cls,
        *,
        names: Mapping[str, str] = {
            "EnergyResolutionSigmaRelABC": "sigma_rel",
            "EnergyResolutionMatrixBC": "matrix",
            "Edges": "e_edges",
            "BinCenter": "e_bincenter",
        },
        path: str | None = None,
        labels: Mapping = {},
        replicate_outputs: tuple[KeyLike, ...] = ((),),
    ) -> tuple[EnergyResolution, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        instance = cls(bare=True)
        key_EnergyResolutionMatrixBC = (names["EnergyResolutionMatrixBC"],)
        key_EnergyResolutionSigmaRelABC = (names["EnergyResolutionSigmaRelABC"],)
        key_BinCenter = (names["BinCenter"],)
        key_Edges = (names["Edges"],)
        if path:
            tpath = tuple(path.split("."))
            key_EnergyResolutionMatrixBC = tpath + key_EnergyResolutionMatrixBC
            key_EnergyResolutionSigmaRelABC = tpath + key_EnergyResolutionSigmaRelABC
            key_BinCenter = tpath + key_BinCenter
            key_Edges = tpath + key_Edges

        _EnergyResolutionSigmaRelABC = instance.add_EnergyResolutionSigmaRelABC(
            names["EnergyResolutionSigmaRelABC"],
            labels.get("EnergyResolutionSigmaRelABC", {}),
        )
        nodes[key_EnergyResolutionSigmaRelABC] = _EnergyResolutionSigmaRelABC
        for iname, input in _EnergyResolutionSigmaRelABC.inputs.iter_kw_items():
            inputs[key_EnergyResolutionSigmaRelABC + (iname,)] = input
        outputs[key_EnergyResolutionSigmaRelABC] = _EnergyResolutionSigmaRelABC.outputs["RelSigma"]

        _BinCenter = instance.add_BinCenter("BinCenter", labels.get("BinCenter", {}))
        nodes[key_BinCenter] = _BinCenter
        inputs[key_Edges] = _BinCenter.inputs[0]
        outputs[key_BinCenter] = (out_bincenter := _BinCenter.outputs[0])

        out_relsigma = _EnergyResolutionSigmaRelABC.outputs["RelSigma"]
        out_bincenter >> _EnergyResolutionSigmaRelABC.inputs["Energy"]

        label_int = labels.get("EnergyResolution", {})
        for key in replicate_outputs:
            if isinstance(key, str):
                key = (key,)
            name = ".".join(key_EnergyResolutionMatrixBC + key)
            eres = instance.add_EnergyResolutionMatrixBC(name, label_int)
            nodes[key_EnergyResolutionMatrixBC + key] = eres
            inputs[key_EnergyResolutionMatrixBC + key] = eres.inputs["Edges"]
            outputs[key_EnergyResolutionMatrixBC + key] = eres.outputs[0]

            out_relsigma >> eres.inputs["RelSigma"]

        NodeStorage.update_current(storage, strict=True)
        return instance, storage
