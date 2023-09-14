from kedro.io import AbstractDataSet
import torch
from pathlib import Path
from typing import Any, Dict


class PytorchDatasetModel(AbstractDataSet):
    def __init__(self,filepath:str, model: torch.nn.Module = None):
        self._model = model
        self._filepath = Path(filepath)

    def _load(self) -> Dict:
        model = torch.load(self._filepath)
        return model

    def _save(self,model:torch.nn.Module) -> None:
        torch.save(model, self._filepath)

    def _describe(self) -> dict[str, Any]:
        return dict(filepath=self._filepath)
class PytorchDatasetDict(AbstractDataSet):
    def __init__(self,filepath:str):
        self._filepath = Path(filepath)

    def _load(self) -> Dict:
        model = torch.load(self._filepath)
        return model

    def _save(self,data: Dict) -> None:
        torch.save(data, self._filepath)

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)