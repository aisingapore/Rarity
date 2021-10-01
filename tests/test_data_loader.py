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

from typing import List
import pandas as pd
import pytest


# test cases for CSVDataLoader
def test_CSVDataLoader_correct_input_single_vs_bimodal(csv_loader_single_modal_cls, csv_loader_bimodal_cls):
    csv_loader_single_modal_cls.get_yPreds = csv_loader_bimodal_cls.get_yPreds
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value == 'no. of yPred_files must be equal to no. of model_names in correct order'


def test_CSVDataLoader_correct_input_analysis_type(csv_loader_single_modal_cls):
    csv_loader_single_modal_cls.get_analysis_type = 'prediction'
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value == "Currently supported analysis types: \
                                ['Regression', 'Binary-Classification', 'Multiclass-Classification']"


def test_CSVDataLoader_correct_input_binary(csv_loader_bimodal_cls, csv_loader_bimodal_cls_multi):
    csv_loader_bimodal_cls.get_yTrue = csv_loader_bimodal_cls_multi.get_yTrue
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value == "Data for yTrue_file doesn't seem to be a binary-class prediction. \
                                Please ensure yTrue_file consists of array of only 2 unique class"


def test_CSVDataLoader_correct_input_multiclass(csv_loader_single_modal_cls, csv_loader_bimodal_cls):
    for loader in [csv_loader_single_modal_cls, csv_loader_bimodal_cls]:
        loader.get_analysis_type = 'multiclass-classification'
        with pytest.raises(AssertionError) as excinfo:
            assert excinfo.value == "Data for yPred_file doesn't seem to be a multiclass prediction. \
                                    Please ensure there is prediction probabilities for each class in the yPred file."


def test_CSVDataLoader_get_features(csv_loader_single_modal_cls, csv_loader_bimodal_cls):
    assert isinstance(csv_loader_single_modal_cls.get_features(), pd.DataFrame)
    assert isinstance(csv_loader_bimodal_cls.get_features(), pd.DataFrame)


def test_CSVDataLoader_get_yTrue(csv_loader_single_modal_cls, csv_loader_bimodal_cls):
    assert isinstance(csv_loader_single_modal_cls.get_yTrue(), pd.DataFrame)
    assert csv_loader_single_modal_cls.get_yTrue().columns[0] == 'yTrue'
    assert isinstance(csv_loader_bimodal_cls.get_yTrue(), pd.DataFrame)
    assert csv_loader_bimodal_cls.get_yTrue().columns[0] == 'yTrue'


def test_CSVDataLoader_get_yPreds(csv_loader_single_modal_reg, csv_loader_single_modal_cls, 
                                csv_loader_bimodal_reg, csv_loader_bimodal_cls):
    for ana_type in [csv_loader_single_modal_reg.get_analysis_type(), 
                    csv_loader_single_modal_cls.get_analysis_type(), 
                    csv_loader_bimodal_reg.get_analysis_type(), 
                    csv_loader_bimodal_cls.get_analysis_type()]:
        if ana_type == 'regression':
            assert isinstance(csv_loader_single_modal_reg.get_yPreds(), pd.DataFrame)
            assert all('yPred_' in col for col in csv_loader_single_modal_reg.get_yPreds().columns)
            assert isinstance(csv_loader_bimodal_reg.get_yPreds(), pd.DataFrame)
            assert all('yPred_' in col for col in csv_loader_bimodal_reg.get_yPreds().columns)

        elif 'classification' in ana_type:
            cols = ['model', 'yPred-label']
            assert isinstance(csv_loader_single_modal_cls.get_yPreds(), List)
            assert all(col in list(csv_loader_single_modal_cls.get_yPreds()[0].columns) for col in cols)

            assert isinstance(csv_loader_bimodal_cls.get_yPreds(), List)
            assert all(col in list(csv_loader_bimodal_cls.get_yPreds()[0].columns) for col in cols)
            assert all(col in list(csv_loader_bimodal_cls.get_yPreds()[1].columns) for col in cols)


def test_CSVDataLoader_get_model_list(csv_loader_single_modal_cls, csv_loader_bimodal_cls):
    assert isinstance(csv_loader_single_modal_cls.get_model_list(), List)
    assert len(csv_loader_single_modal_cls.get_model_list()) == 1
    assert isinstance(csv_loader_bimodal_cls.get_model_list(), List)
    assert len(csv_loader_bimodal_cls.get_model_list()) == 2


def test_CSVDataLoader_get_analysis_type(csv_loader_single_modal_cls, csv_loader_bimodal_cls):
    analysis_types = ['regression', 'binary-classification', 'multiclass-classification']
    assert csv_loader_single_modal_cls.get_analysis_type() in analysis_types
    assert csv_loader_bimodal_cls.get_analysis_type() in analysis_types


def test_CSVDataLoader_get_all(csv_loader_single_modal_reg, csv_loader_single_modal_cls, 
                            csv_loader_bimodal_reg, csv_loader_bimodal_cls):
    for ana_type in [csv_loader_single_modal_reg.get_analysis_type(), 
                    csv_loader_single_modal_cls.get_analysis_type(), 
                    csv_loader_bimodal_reg.get_analysis_type(), 
                    csv_loader_bimodal_cls.get_analysis_type()]:
        if ana_type == 'regression':
            assert isinstance(csv_loader_single_modal_reg.get_all(), pd.DataFrame)
            assert isinstance(csv_loader_bimodal_reg.get_all(), pd.DataFrame)
        elif 'classification' in ana_type:
            assert isinstance(csv_loader_single_modal_cls.get_all(), List)
            assert isinstance(csv_loader_bimodal_cls.get_all(), List)


# test cases for DataframeLoader
def test_DataframeLoader_correct_input_single_vs_bimodal(dataframe_loader_single_modal_cls, 
                                                        dataframe_loader_bimodal_cls):
    dataframe_loader_single_modal_cls.get_yPreds = dataframe_loader_bimodal_cls.get_yPreds
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value == 'no. of yPred_files must be equal to no. of model_names in correct order'


def test_DataframeLoader_correct_input_analysis_type(dataframe_loader_single_modal_cls):
    dataframe_loader_single_modal_cls.get_analysis_type = 'prediction'
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value == "Currently supported analysis types: \
                                ['Regression', 'Binary-Classification', 'Multiclass-Classification']"


def test_DataframeLoader_correct_input_binary(dataframe_loader_bimodal_cls, dataframe_loader_bimodal_cls_multi):
    dataframe_loader_bimodal_cls.get_yTrue = dataframe_loader_bimodal_cls_multi.get_yTrue
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value == "Data for df_yTrue doesn't seem to be a binary-class prediction. \
                                Please ensure df_yTrue consists of data with only 2 unique class"


def test_DataframeLoader_correct_input_multiclass(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    for loader in [dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls]:
        loader.get_analysis_type = 'multiclass-classification'
        with pytest.raises(AssertionError) as excinfo:
            assert excinfo.value == "Data for df_yPred_list doesn't seem to be a multiclass prediction. \
                                    Please ensure there is prediction probabilities for each class \
                                    in the df of df_yPred_list."


def test_DataframeLoader_get_features(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert isinstance(dataframe_loader_single_modal_cls.get_features(), pd.DataFrame)
    assert isinstance(dataframe_loader_bimodal_cls.get_features(), pd.DataFrame)


def test_DataframeLoader_get_yTrue(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert isinstance(dataframe_loader_single_modal_cls.get_yTrue(), pd.DataFrame)
    assert dataframe_loader_single_modal_cls.get_yTrue().columns[0] == 'yTrue'
    assert isinstance(dataframe_loader_bimodal_cls.get_yTrue(), pd.DataFrame)
    assert dataframe_loader_bimodal_cls.get_yTrue().columns[0] == 'yTrue'


def test_DataframeLoader_get_yPreds(dataframe_loader_single_modal_reg, dataframe_loader_single_modal_cls, 
                                    dataframe_loader_bimodal_reg, dataframe_loader_bimodal_cls):
    for ana_type in [dataframe_loader_single_modal_reg.get_analysis_type(), 
                    dataframe_loader_single_modal_cls.get_analysis_type(), 
                    dataframe_loader_bimodal_reg.get_analysis_type(), 
                    dataframe_loader_bimodal_cls.get_analysis_type()]:
        if ana_type == 'regression':
            assert isinstance(dataframe_loader_single_modal_reg.get_yPreds(), pd.DataFrame)
            assert all('yPred_' in col for col in dataframe_loader_single_modal_reg.get_yPreds().columns)
            assert isinstance(dataframe_loader_bimodal_reg.get_yPreds(), pd.DataFrame)
            assert all('yPred_' in col for col in dataframe_loader_bimodal_reg.get_yPreds().columns)

        elif 'classification' in ana_type:
            assert isinstance(dataframe_loader_single_modal_cls.get_yPreds(), List)
            cols = ['model', 'yPred-label']
            assert all(col in list(dataframe_loader_single_modal_cls.get_yPreds()[0].columns) for col in cols)

            assert isinstance(dataframe_loader_bimodal_cls.get_yPreds(), List)
            assert all(col in list(dataframe_loader_bimodal_cls.get_yPreds()[0].columns) for col in cols)
            assert all(col in list(dataframe_loader_bimodal_cls.get_yPreds()[1].columns) for col in cols)


def test_DataframeLoader_get_model_list(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert isinstance(dataframe_loader_single_modal_cls.get_model_list(), List)
    assert len(dataframe_loader_single_modal_cls.get_model_list()) == 1
    assert isinstance(dataframe_loader_bimodal_cls.get_model_list(), List)
    assert len(dataframe_loader_bimodal_cls.get_model_list()) == 2


def test_DataframeLoader_get_analysis_type(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    analysis_types = ['regression', 'binary-classification', 'multiclass-classification']
    assert dataframe_loader_single_modal_cls.get_analysis_type() in analysis_types
    assert dataframe_loader_bimodal_cls.get_analysis_type() in analysis_types


def test_DataframeLoader_get_all(dataframe_loader_single_modal_reg, dataframe_loader_single_modal_cls, 
                                dataframe_loader_bimodal_reg, dataframe_loader_bimodal_cls):
    for ana_type in [dataframe_loader_single_modal_reg.get_analysis_type(), 
                    dataframe_loader_single_modal_cls.get_analysis_type(), 
                    dataframe_loader_bimodal_reg.get_analysis_type(), 
                    dataframe_loader_bimodal_cls.get_analysis_type()]:
        if ana_type == 'regression':
            assert isinstance(dataframe_loader_single_modal_reg.get_all(), pd.DataFrame)
            assert isinstance(dataframe_loader_bimodal_reg.get_all(), pd.DataFrame)
        elif 'classification' in ana_type:
            assert isinstance(dataframe_loader_single_modal_cls.get_all(), List)
            assert isinstance(dataframe_loader_bimodal_cls.get_all(), List)
