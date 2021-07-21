import os
from typing import List
import pandas as pd
import pytest
from tenjin.data_loader import CSVDataLoader, DataframeLoader

# test cases for CSVDataLoader
def test_CSVDataLoader_correct_input_single_vs_bimodal(csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls):
    csv_data_loader_single_modal_cls.get_yPreds = csv_data_loader_bimodal_cls.get_yPreds
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value ==  'no. of yPred_files must be equal to no. of model_names in correct order'

def test_CSVDataLoader_correct_input_analysis_type(csv_data_loader_single_modal_cls):
    csv_data_loader_single_modal_cls.get_analysis_type = 'prediction'
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value ==  "Currently supported analysis types: ['Regression', 'Binary-Classification', 'Multiclass-Classification']"

def test_CSVDataLoader_correct_input_binary(csv_data_loader_bimodal_cls, csv_data_loader_bimodal_cls_multi):
    csv_data_loader_bimodal_cls.get_yTrue = csv_data_loader_bimodal_cls_multi.get_yTrue
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value ==  "Data for yTrue_file doesn't seem to be a binary-class prediction. Please ensure yTrue_file consists of array of only 2 unique class"

def test_CSVDataLoader_correct_input_multiclass(csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls):
    for loader in [csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls]:
        loader.get_analysis_type = 'multiclass-classification'
        with pytest.raises(AssertionError) as excinfo:
            assert excinfo.value == "Data for yPred_file doesn't seem to be a multiclass prediction. Please ensure there is prediction probabilities for each class in the yPred file."

def test_CSVDataLoader_get_features(csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls):
    assert isinstance(csv_data_loader_single_modal_cls.get_features(), pd.DataFrame)
    assert isinstance(csv_data_loader_bimodal_cls.get_features(), pd.DataFrame)

def test_CSVDataLoader_get_yTrue(csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls):
    assert isinstance(csv_data_loader_single_modal_cls.get_yTrue(), pd.DataFrame)
    assert csv_data_loader_single_modal_cls.get_yTrue().columns[0] == 'yTrue'
    assert isinstance(csv_data_loader_bimodal_cls.get_yTrue(), pd.DataFrame)
    assert csv_data_loader_bimodal_cls.get_yTrue().columns[0] == 'yTrue'

def test_CSVDataLoader_get_yPreds(csv_data_loader_single_modal_reg, csv_data_loader_single_modal_cls, csv_data_loader_bimodal_reg, csv_data_loader_bimodal_cls):
    for ana_type in [csv_data_loader_single_modal_reg.get_analysis_type(), 
                    csv_data_loader_single_modal_cls.get_analysis_type(), 
                    csv_data_loader_bimodal_reg.get_analysis_type(), 
                    csv_data_loader_bimodal_cls.get_analysis_type()]:
        if ana_type=='regression':
            assert isinstance(csv_data_loader_single_modal_reg.get_yPreds(), pd.DataFrame)
            assert all( 'yPred_' in col for col in csv_data_loader_single_modal_reg.get_yPreds().columns)
            assert isinstance(csv_data_loader_bimodal_reg.get_yPreds(), pd.DataFrame)
            assert all( 'yPred_' in col for col in csv_data_loader_bimodal_reg.get_yPreds().columns)

        elif 'classification' in ana_type:
            assert isinstance(csv_data_loader_single_modal_cls.get_yPreds(), List)
            assert all(col in list(csv_data_loader_single_modal_cls.get_yPreds()[0].columns) for col in ['model', 'yPred-label'])

            assert isinstance(csv_data_loader_bimodal_cls.get_yPreds(), List)
            assert all(col in list(csv_data_loader_bimodal_cls.get_yPreds()[0].columns) for col in ['model', 'yPred-label'])
            assert all(col in list(csv_data_loader_bimodal_cls.get_yPreds()[1].columns) for col in ['model', 'yPred-label'] )

def test_CSVDataLoader_get_model_list(csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls):
    assert isinstance(csv_data_loader_single_modal_cls.get_model_list(), List)
    assert len(csv_data_loader_single_modal_cls.get_model_list())==1
    assert isinstance(csv_data_loader_bimodal_cls.get_model_list(), List)
    assert len(csv_data_loader_bimodal_cls.get_model_list())==2

def test_CSVDataLoader_get_analysis_type(csv_data_loader_single_modal_cls, csv_data_loader_bimodal_cls):
    assert csv_data_loader_single_modal_cls.get_analysis_type() in ['regression', 'binary-classification', 'multiclass-classification']
    assert csv_data_loader_bimodal_cls.get_analysis_type() in ['regression', 'binary-classification', 'multiclass-classification']

def test_CSVDataLoader_get_all(csv_data_loader_single_modal_reg, csv_data_loader_single_modal_cls, csv_data_loader_bimodal_reg, csv_data_loader_bimodal_cls):
    for ana_type in [csv_data_loader_single_modal_reg.get_analysis_type(), 
                    csv_data_loader_single_modal_cls.get_analysis_type(), 
                    csv_data_loader_bimodal_reg.get_analysis_type(), 
                    csv_data_loader_bimodal_cls.get_analysis_type()]:
        if ana_type=='regression':
            assert isinstance(csv_data_loader_single_modal_reg.get_all(), pd.DataFrame)
            assert isinstance(csv_data_loader_bimodal_reg.get_all(), pd.DataFrame)
        elif 'classification' in ana_type:
            assert isinstance(csv_data_loader_single_modal_cls.get_all(), List)
            assert isinstance(csv_data_loader_bimodal_cls.get_all(), List)


# test cases for DataframeLoader
def test_DataframeLoader_correct_input_single_vs_bimodal(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    dataframe_loader_single_modal_cls.get_yPreds = dataframe_loader_bimodal_cls.get_yPreds
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value ==  'no. of yPred_files must be equal to no. of model_names in correct order'

def test_DataframeLoader_correct_input_analysis_type(dataframe_loader_single_modal_cls):
    dataframe_loader_single_modal_cls.get_analysis_type = 'prediction'
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value ==  "Currently supported analysis types: ['Regression', 'Binary-Classification', 'Multiclass-Classification']"

def test_DataframeLoader_correct_input_binary(dataframe_loader_bimodal_cls, dataframe_loader_bimodal_cls_multi):
    dataframe_loader_bimodal_cls.get_yTrue = dataframe_loader_bimodal_cls_multi.get_yTrue
    with pytest.raises(AssertionError) as excinfo:
        assert excinfo.value ==  "Data for df_yTrue doesn't seem to be a binary-class prediction. Please ensure df_yTrue consists of data with only 2 unique class"

def test_DataframeLoader_correct_input_multiclass(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    for loader in [dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls]:
        loader.get_analysis_type = 'multiclass-classification'
        with pytest.raises(AssertionError) as excinfo:
            assert excinfo.value == "Data for df_yPred_list doesn't seem to be a multiclass prediction. Please ensure there is prediction probabilities for each class in the df of df_yPred_list."

def test_DataframeLoader_get_features(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert isinstance(dataframe_loader_single_modal_cls.get_features(), pd.DataFrame)
    assert isinstance(dataframe_loader_bimodal_cls.get_features(), pd.DataFrame)

def test_DataframeLoader_get_yTrue(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert isinstance(dataframe_loader_single_modal_cls.get_yTrue(), pd.DataFrame)
    assert dataframe_loader_single_modal_cls.get_yTrue().columns[0] == 'yTrue'
    assert isinstance(dataframe_loader_bimodal_cls.get_yTrue(), pd.DataFrame)
    assert dataframe_loader_bimodal_cls.get_yTrue().columns[0] == 'yTrue'

def test_DataframeLoader_get_yPreds(dataframe_loader_single_modal_reg, dataframe_loader_single_modal_cls, dataframe_loader_bimodal_reg, dataframe_loader_bimodal_cls):
    for ana_type in [dataframe_loader_single_modal_reg.get_analysis_type(), 
                    dataframe_loader_single_modal_cls.get_analysis_type(), 
                    dataframe_loader_bimodal_reg.get_analysis_type(), 
                    dataframe_loader_bimodal_cls.get_analysis_type()]:
        if ana_type=='regression':
            assert isinstance(dataframe_loader_single_modal_reg.get_yPreds(), pd.DataFrame)
            assert all( 'yPred_' in col for col in dataframe_loader_single_modal_reg.get_yPreds().columns)
            assert isinstance(dataframe_loader_bimodal_reg.get_yPreds(), pd.DataFrame)
            assert all( 'yPred_' in col for col in dataframe_loader_bimodal_reg.get_yPreds().columns)

        elif 'classification' in ana_type:
            assert isinstance(dataframe_loader_single_modal_cls.get_yPreds(), List)
            assert all(col in list(dataframe_loader_single_modal_cls.get_yPreds()[0].columns) for col in ['model', 'yPred-label'])

            assert isinstance(dataframe_loader_bimodal_cls.get_yPreds(), List)
            assert all(col in list(dataframe_loader_bimodal_cls.get_yPreds()[0].columns) for col in ['model', 'yPred-label'])
            assert all(col in list(dataframe_loader_bimodal_cls.get_yPreds()[1].columns) for col in ['model', 'yPred-label'] )

def test_DataframeLoader_get_model_list(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert isinstance(dataframe_loader_single_modal_cls.get_model_list(), List)
    assert len(dataframe_loader_single_modal_cls.get_model_list())==1
    assert isinstance(dataframe_loader_bimodal_cls.get_model_list(), List)
    assert len(dataframe_loader_bimodal_cls.get_model_list())==2

def test_DataframeLoader_get_analysis_type(dataframe_loader_single_modal_cls, dataframe_loader_bimodal_cls):
    assert dataframe_loader_single_modal_cls.get_analysis_type() in ['regression', 'binary-classification', 'multiclass-classification']
    assert dataframe_loader_bimodal_cls.get_analysis_type() in ['regression', 'binary-classification', 'multiclass-classification']

def test_DataframeLoader_get_all(dataframe_loader_single_modal_reg, dataframe_loader_single_modal_cls, dataframe_loader_bimodal_reg, dataframe_loader_bimodal_cls):
    for ana_type in [dataframe_loader_single_modal_reg.get_analysis_type(), 
                    dataframe_loader_single_modal_cls.get_analysis_type(), 
                    dataframe_loader_bimodal_reg.get_analysis_type(), 
                    dataframe_loader_bimodal_cls.get_analysis_type()]:
        if ana_type=='regression':
            assert isinstance(dataframe_loader_single_modal_reg.get_all(), pd.DataFrame)
            assert isinstance(dataframe_loader_bimodal_reg.get_all(), pd.DataFrame)
        elif 'classification' in ana_type:
            assert isinstance(dataframe_loader_single_modal_cls.get_all(), List)
            assert isinstance(dataframe_loader_bimodal_cls.get_all(), List)