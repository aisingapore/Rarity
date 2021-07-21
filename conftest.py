import pytest
import os
import pandas as pd
from tenjin.data_loader import CSVDataLoader, DataframeLoader


@pytest.fixture
def csv_loader_single_modal_reg():
    SAMPLE_DATA_DIR = "src/tenjin/examples/sample_data/structured_data/housing_price_prediction/"
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'features_housing.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'yPreds_lasso.csv')
    MODEL_NAMES = ['lasso']
    ANALYSIS_TYPE = 'Regression'

    data_loader = CSVDataLoader(FEATURES_FILE, 
                                Y_TRUE_FILE, 
                                yPred_file_list=[Y_PRED_FILE_1], 
                                model_names_list=MODEL_NAMES, 
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_single_modal_cls():
    SAMPLE_DATA_DIR = "src/tenjin/examples/sample_data/structured_data/income_prediction/"
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'features_income.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'yTrue_income.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'yPred_income_modelA.csv')
    MODEL_NAMES = ['model_A']
    ANALYSIS_TYPE = 'Binary-Classification'

    data_loader = CSVDataLoader(FEATURES_FILE, 
                                Y_TRUE_FILE, 
                                yPred_file_list=[Y_PRED_FILE_1], 
                                model_names_list=MODEL_NAMES, 
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_bimodal_reg():
    SAMPLE_DATA_DIR = "src/tenjin/examples/sample_data/structured_data/housing_price_prediction/"
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'features_housing.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'yPreds_lasso.csv')
    Y_PRED_FILE_2 = os.path.join(SAMPLE_DATA_DIR, 'yPreds_rf.csv')
    MODEL_NAMES = ['lasso', 'random_forest']
    ANALYSIS_TYPE = 'Regression'

    data_loader = CSVDataLoader(FEATURES_FILE, 
                                Y_TRUE_FILE, 
                                yPred_file_list=[Y_PRED_FILE_1, Y_PRED_FILE_2], 
                                model_names_list=MODEL_NAMES, 
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_bimodal_cls():
    SAMPLE_DATA_DIR = "src/tenjin/examples/sample_data/structured_data/income_prediction/"
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'features_income.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'yTrue_income.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'yPred_income_modelA.csv')
    Y_PRED_FILE_2 = os.path.join(SAMPLE_DATA_DIR, 'yPred_income_modelB.csv')
    MODEL_NAMES = ['model_A', 'model_B']
    ANALYSIS_TYPE = 'Binary-Classification'

    data_loader = CSVDataLoader(FEATURES_FILE, 
                                Y_TRUE_FILE, 
                                yPred_file_list=[Y_PRED_FILE_1, Y_PRED_FILE_2], 
                                model_names_list=MODEL_NAMES, 
                                analysis_type=ANALYSIS_TYPE)
    return data_loader


@pytest.fixture
def csv_loader_bimodal_cls_multi():
    SAMPLE_DATA_DIR = "src/tenjin/examples/sample_data/glass_dataset/"
    FEATURES_FILE = os.path.join(SAMPLE_DATA_DIR, 'glass_xfeatures.csv')
    Y_TRUE_FILE = os.path.join(SAMPLE_DATA_DIR, 'glass_yTrue.csv')
    Y_PRED_FILE_1 = os.path.join(SAMPLE_DATA_DIR, 'glass_yPreds_logreg.csv')
    Y_PRED_FILE_2 = os.path.join(SAMPLE_DATA_DIR, 'glass_yPreds_rf.csv')
    MODEL_NAMES = ['logreg', 'rf']
    ANALYSIS_TYPE = 'Multiclass-Classification'

    data_loader = CSVDataLoader(FEATURES_FILE, 
                                Y_TRUE_FILE, 
                                yPred_file_list=[Y_PRED_FILE_1, Y_PRED_FILE_2], 
                                model_names_list=MODEL_NAMES, 
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
                                df_yPred_list=[DF_Y_PRED_1], 
                                model_names_list=MODEL_NAMES, 
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
                                df_yPred_list=[DF_Y_PRED_1], 
                                model_names_list=MODEL_NAMES, 
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
                                df_yPred_list=[DF_Y_PRED_1, DF_Y_PRED_2], 
                                model_names_list=MODEL_NAMES, 
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
                                df_yPred_list=[DF_Y_PRED_1, DF_Y_PRED_2], 
                                model_names_list=MODEL_NAMES, 
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
                                df_yPred_list=[DF_Y_PRED_1, DF_Y_PRED_2], 
                                model_names_list=MODEL_NAMES, 
                                analysis_type=ANALYSIS_TYPE)
    return data_loader
