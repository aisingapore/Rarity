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

import os
import sys

from configs import configs
from rarity import GapAnalyzer
from rarity.data_loader import CSVDataLoader


CONFIGS_DIR = 'configs'
xFEATURES_FILE = os.path.join(CONFIGS_DIR, configs.XFEATURE_FILEPATH)
yTRUE_FILE = os.path.join(CONFIGS_DIR, configs.YTRUE_FILEPATH)
yPRED_FILE_LIST = [os.path.join(CONFIGS_DIR, file) for file in configs.YPRED_FILEPATH]


def main():
    data_loader = CSVDataLoader(xFEATURES_FILE, yTRUE_FILE, yPRED_FILE_LIST, configs.MODEL_NAME_LIST, configs.ANALYSIS_TYPE)
    analyzer = GapAnalyzer(data_loader, configs.ANALYSIS_TITLE, configs.PORT)
    analyzer.run()


if __name__ == '__main__':
    if len(os.listdir('configs/csv_data/')) == 1:
        sys.exit(
            """
            'configs/csv_data/' directory is empty. Please place xFeature, yTrue, yPred files into the mentioned folder 
            and update 'configs/configs.py' with the correct analysis base info.
            """)
    else:
        main()
