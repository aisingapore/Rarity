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

import pytest
import os
import pandas as pd
from rarity.data_loader import CSVDataLoader, DataframeLoader
# add this in the conftest.py under tests folder
from selenium.webdriver.chrome.options import Options


def pytest_setup_options():
    options = Options()

    # added mainly for integration test in gitlab-ci to resolve 
    # (unknown error: DevToolsActivePort file doesn't exist)
    # (The process started from chrome location /usr/bin/google-chrome is no longer running, 
    # so ChromeDriver is assuming that Chrome has crashed.)
    # solution reference => https://github.com/plotly/dash/issues/1420
    options.add_argument('--no-sandbox')
    return options


@pytest.fixture
def csv_loader_single_modal_reg():
    SAMPLE_DATA_DIR = './tests/sample_data/regression/'
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'reg_features.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'reg_yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'reg_yPreds_modelA.csv')
    MODEL_NAMES = ['model_A']
    ANALYSIS_TYPE = 'Regression'

    data_loader = CSVDataLoader(FEATURES_FILE,
                                Y_TRUE_FILE,
                                yPred_file_ls=[Y_PRED_FILE_1],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_single_modal_cls():
    SAMPLE_DATA_DIR = './tests/sample_data/classification/binary/'
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'binary_features.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'binary_yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'binary_yPreds_modelA.csv')
    MODEL_NAMES = ['model_A']
    ANALYSIS_TYPE = 'Binary-Classification'

    data_loader = CSVDataLoader(FEATURES_FILE,
                                Y_TRUE_FILE,
                                yPred_file_ls=[Y_PRED_FILE_1],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_bimodal_reg():
    SAMPLE_DATA_DIR = './tests/sample_data/regression/'
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'reg_features.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'reg_yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'reg_yPreds_modelA.csv')
    Y_PRED_FILE_2 = os.path.join(SAMPLE_DATA_DIR, 'reg_yPreds_modelB.csv')
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Regression'

    data_loader = CSVDataLoader(FEATURES_FILE,
                                Y_TRUE_FILE,
                                yPred_file_ls=[Y_PRED_FILE_1, Y_PRED_FILE_2],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_bimodal_cls():
    SAMPLE_DATA_DIR = './tests/sample_data/classification/binary/'
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'binary_features.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'binary_yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'binary_yPreds_modelA.csv')
    Y_PRED_FILE_2 = os.path.join(SAMPLE_DATA_DIR, 'binary_yPreds_modelB.csv')
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Binary-Classification'

    data_loader = CSVDataLoader(FEATURES_FILE,
                                Y_TRUE_FILE,
                                yPred_file_ls=[Y_PRED_FILE_1, Y_PRED_FILE_2],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_bimodal_cls_multi():
    SAMPLE_DATA_DIR = './tests/sample_data/classification/multiclass/'
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'multiclass_features.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'multiclass_yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'multiclass_yPreds_modelA.csv')
    Y_PRED_FILE_2 = os.path.join(SAMPLE_DATA_DIR, 'multiclass_yPreds_modelB.csv')
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Multiclass-Classification'

    data_loader = CSVDataLoader(FEATURES_FILE,
                                Y_TRUE_FILE,
                                yPred_file_ls=[Y_PRED_FILE_1, Y_PRED_FILE_2],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def dataframe_loader_single_modal_reg():
    DF_FEATURES = pd.DataFrame([[0.1, 2.5, 3.6], [0.5, 2.2, 6.6]], columns=['x1', 'x2', 'x3'])
    DF_Y_TRUE = pd.DataFrame([[22.6], [36.6]], columns=['actual'])
    DF_Y_PRED_1 = pd.DataFrame([[22.2], [35.0]], columns=['pred'])
    MODEL_NAMES = ['model_A']
    ANALYSIS_TYPE = 'Regression'

    data_loader = DataframeLoader(DF_FEATURES,
                                DF_Y_TRUE,
                                df_yPred_ls=[DF_Y_PRED_1],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def dataframe_loader_single_modal_cls():
    DF_FEATURES = pd.DataFrame([[0.1, 2.5, 3.6], [0.5, 2.2, 6.6], [0.3, 2.3, 5.2]], columns=['x1', 'x2', 'x3'])
    DF_Y_TRUE = pd.DataFrame([[0], [1], [1]], columns=['actual'])
    DF_Y_PRED_1 = pd.DataFrame([[0.38, 0.62], [0.86, 0.14], [0.78, 0.22]], columns=['0', '1'])
    MODEL_NAMES = ['model_A']
    ANALYSIS_TYPE = 'Binary-Classification'

    data_loader = DataframeLoader(DF_FEATURES,
                                DF_Y_TRUE,
                                df_yPred_ls=[DF_Y_PRED_1],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def dataframe_loader_bimodal_reg():
    DF_FEATURES = pd.DataFrame([[0.1, 2.5, 3.6], [0.5, 2.2, 6.6]], columns=['x1', 'x2', 'x3'])
    DF_Y_TRUE = pd.DataFrame([[22.6], [36.6]], columns=['actual'])
    DF_Y_PRED_1 = pd.DataFrame([[22.2], [35.0]], columns=['pred'])
    DF_Y_PRED_2 = pd.DataFrame([[22.2], [35.0]], columns=['pred'])
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Regression'

    data_loader = DataframeLoader(DF_FEATURES,
                                DF_Y_TRUE,
                                df_yPred_ls=[DF_Y_PRED_1, DF_Y_PRED_2],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def dataframe_loader_bimodal_cls():
    DF_FEATURES = pd.DataFrame([[0.1, 2.5, 3.6], [0.5, 2.2, 6.6]], columns=['x1', 'x2', 'x3'])
    DF_Y_TRUE = pd.DataFrame([[0], [1]], columns=['actual'])
    DF_Y_PRED_1 = pd.DataFrame([[0.38, 0.62], [0.86, 0.14]], columns=['0', '1'])
    DF_Y_PRED_2 = pd.DataFrame([[0.56, 0.44], [0.68, 0.32]], columns=['0', '1'])
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Binary-Classification'

    data_loader = DataframeLoader(DF_FEATURES,
                                DF_Y_TRUE,
                                df_yPred_ls=[DF_Y_PRED_1, DF_Y_PRED_2],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def dataframe_loader_bimodal_cls_multi():
    DF_FEATURES = pd.DataFrame([[0.1, 2.5, 3.6], [0.5, 2.2, 6.6], [0.3, 2.3, 5.2]], columns=['x1', 'x2', 'x3'])
    DF_Y_TRUE = pd.DataFrame([[0], [1], [2]], columns=['actual'])
    DF_Y_PRED_1 = pd.DataFrame([[0.12, 0.28, 0.60], [0.80, 0.16, 0.04], [0.2, 0.7, 0.1]], columns=['0', '1', '2'])
    DF_Y_PRED_2 = pd.DataFrame([[0.60, 0.30, 0.10], [0.70, 0.22, 0.08], [0.05, 0.15, 0.80]], columns=['0', '1', '2'])
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Multiclass-Classification'

    data_loader = DataframeLoader(DF_FEATURES,
                                DF_Y_TRUE,
                                df_yPred_ls=[DF_Y_PRED_1, DF_Y_PRED_2],
                                model_names_ls=MODEL_NAMES,
                                analysis_type=ANALYSIS_TYPE)
    return data_loader
