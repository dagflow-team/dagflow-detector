from __future__ import annotations

from typing import TYPE_CHECKING, List, Mapping, Optional, Tuple

from dagflow.exception import ConnectionError
from dagflow.lib import VectorMatrixProduct
from dagflow.metanode import MetaNode
from dagflow.storage import NodeStorage
from multikeydict.typing import KeyLike

from detector.RebinMatrix import RebinMatrix, RebinModesType

if TYPE_CHECKING:
    from dagflow.node import Node


class Rebin(MetaNode):
    __slots__ = ("_RebinMatrixList", "_VectorMatrixProductList")

    _RebinMatrixList: List["Node"]
    _VectorMatrixProductList: List["Node"]

    def __init__(
        self,
        *,
        bare: bool = False,
        mode: RebinModesType = "numba",
        atol: float = 0.0,
        rtol: float = 1e-14,
        labels: Mapping = {},
    ):
        super().__init__()
        self._RebinMatrixList = []
        self._VectorMatrixProductList = []
        if bare:
            return

        self.add_RebinMatrix(
            name="RebinMatrix",
            mode=mode,
            atol=atol,
            rtol=rtol,
            label=labels.get("RebinMatrix", {}),
        )
        self.add_VectorMatrixProduct("VectorMatrixProduct", labels.get("VectorMatrixProduct", {}))
        self._bind_outputs()

    def add_RebinMatrix(
        self,
        name: str = "RebinMatrix",
        mode: RebinModesType = "numba",
        atol: float = 0.0,
        rtol: float = 1e-14,
        label: Mapping = {},
    ) -> RebinMatrix:
        _RebinMatrix = RebinMatrix(name=name, mode=mode, atol=atol, rtol=rtol, label=label)
        self._RebinMatrixList.append(_RebinMatrix)
        self._add_node(
            _RebinMatrix,
            kw_inputs=["EdgesOld", "EdgesNew"],
            kw_outputs=["Matrix"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return _RebinMatrix

    def add_VectorMatrixProduct(
        self, name: str = "VectorMatrixProduct", label: Mapping = {}
    ) -> VectorMatrixProduct:
        _VectorMatrixProduct = VectorMatrixProduct(name, mode="column", label=label)
        self._VectorMatrixProductList.append(_VectorMatrixProduct)
        self._add_node(
            _VectorMatrixProduct,
            kw_inputs=["vector", "matrix"],
            kw_outputs=["result"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return _VectorMatrixProduct

    def _bind_outputs(self) -> None:
        if (l1 := len(self._VectorMatrixProductList)) != (l2 := len(self._RebinMatrixList)):
            raise ConnectionError(
                "Cannot bind outputs! Nodes must be deuces of (VectorMatrixProduct, RebinMatrix), "
                f"but current lengths are {l1}, {l2}!",
                node=self,
            )
        for _VectorMatrixProduct, _RebinMatrix in zip(
            self._VectorMatrixProductList, self._RebinMatrixList
        ):
            _RebinMatrix.outputs["Matrix"] >> _VectorMatrixProduct.inputs["matrix"]

    @classmethod
    def replicate(
        cls,
        name_RebinMatrix: str = "RebinMatrix",
        name_VectorMatixProduct: str = "VectorMatrixProduct",
        path: Optional[str] = None,
        labels: Mapping = {},
        *,
        replicate: Tuple[KeyLike, ...] = ((),),
    ) -> Tuple["Rebin", "NodeStorage"]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        instance = cls(bare=True)
        key_VectorMatixProduct = (name_VectorMatixProduct,)
        key_RebinMatrix = (name_RebinMatrix,)
        if path:
            tpath = tuple(path.split("."))
            key_VectorMatixProduct = tpath + key_VectorMatixProduct
            key_RebinMatrix = tpath + key_RebinMatrix

        _RebinMatrix = instance.add_RebinMatrix(
            name_RebinMatrix,
            labels.get("RebinMatrix", {}),
        )
        nodes[key_RebinMatrix] = _RebinMatrix
        for iname, input in _RebinMatrix.inputs.iter_kw_items():
            inputs[key_RebinMatrix + (iname,)] = input
        outputs[key_RebinMatrix] = _RebinMatrix.outputs["RebinMatrix"]

        _VectorMatrixProduct = instance.add_VectorMatrixProduct(
            "VectorMatrixProduct", labels.get("VectorMatrixProduct", {})
        )
        nodes[key_VectorMatixProduct] = _VectorMatrixProduct
        for iname, input in _VectorMatrixProduct.inputs.iter_kw_items():
            inputs[key_VectorMatixProduct + (iname,)] = input
        outputs[key_VectorMatixProduct] = _VectorMatrixProduct.outputs["result"]

        _RebinMatrix.outputs["Matrix"] >> _VectorMatrixProduct.inputs["matrix"]

        label_int = labels.get("Rebin", {})
        for key in replicate:
            if isinstance(key, str):
                key = (key,)

            name = ".".join(key_VectorMatixProduct + key)
            _VectorMatrixProduct = instance.add_VectorMatrixProduct(name, label_int)
            nodes[name] = _VectorMatrixProduct
            for iname, input in _VectorMatrixProduct.inputs.iter_kw_items():
                inputs[name + (iname,)] = input
            outputs[name] = _VectorMatrixProduct.outputs["result"]
            _RebinMatrix.outputs["Matrix"] >> _VectorMatrixProduct.inputs["matrix"]

        NodeStorage.update_current(storage, strict=True)
        return instance, storage
