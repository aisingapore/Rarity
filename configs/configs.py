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

# ********************************************************************************************************************************
# To be updated by user accordingly if to run this script using saved csv files
# Replace 'example_xxx.csv' with user's file name
xFeature_filename = 'example_user_xFeatures.csv'
yTrue_filename = 'example_user_yTrue.csv'
# yPred_file can be single file or max 2 files, wrap in a list
yPred_filename = ['example_user_yPred_model_x.csv', 'example_user_yPred_model_y.csv']

# model name must be listed according to the same sequence of the yPred_filename list above
# can be single model or max bi-modal, wrap in a list
MODEL_NAME_LIST = ['example_model_x', 'example_model_y']

# Supported analysis type: 'Regression', 'Binary Classification', 'Multiclass Classification'
ANALYSIS_TYPE = 'Regression'
ANALYSIS_TITLE = 'example_Customer Churn Prediction'

# Defaults to 8000, user can re-define to a new port number of choice
PORT = 8000
# *******************************************************************************************************************************

# No modification from user is needed from this line onwards
DATA_DIR = 'csv_data'
XFEATURE_FILEPATH = os.path.join(DATA_DIR, xFeature_filename)
YTRUE_FILEPATH = os.path.join(DATA_DIR, yTrue_filename)
YPRED_FILEPATH = [os.path.join(DATA_DIR, file) for file in yPred_filename]
