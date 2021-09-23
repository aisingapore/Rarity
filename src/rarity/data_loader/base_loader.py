from abc import ABC, abstractmethod
from typing import List


class BaseLoader(ABC):
    def __init__(
        self,
        xFeatures_file: str,
        yTrue_file: str,
        yPred_file_ls: List[str] = [],
        model_names_ls: List[str] = [],
        analysis_type: str = None
    ):
        self.xFeatures = xFeatures_file
        self.yTrue = yTrue_file
        self.yPreds = yPred_file_ls
        self.models = model_names_ls
        self.analysis_type = analysis_type

    @abstractmethod
    def get_features(self):
        raise NotImplementedError()

    @abstractmethod
    def get_yTrue(self):
        raise NotImplementedError()

    @abstractmethod
    def get_yPreds(self):
        raise NotImplementedError()

    @abstractmethod
    def get_model_list(self):
        raise NotImplementedError()

    @abstractmethod
    def get_analysis_type(self):
        raise NotImplementedError()

    @abstractmethod
    def get_all(self):
        raise NotImplementedError()
