# Copyright 2021 AI Singapore. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
