import numpy as np
import pandas as pd


class IntMissPredictions:
    """
    - Transform raw data into input format suitable for plotting with plotly

    Arguments:
        data_loader {class object} -- class object from data_loader pipeline

    Returns:
        if analysis_type == 'regression'
            df {dataframe} -- dataframe storing info needed for visualizer
        elif analysis_type == 'classification'
            ls_dfs_viz {list} -- list of dataframes for overview
            ls_dfs_by_label {list} -- list of dataframes by individual label class
            ls_dfs_by_label_state {list} -- list of dataframes storing basic stats of each label class
        """
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = self.data_loader.get_analysis_type()
        self.models = self.data_loader.get_model_list()

    def xform(self):
        if self.analysis_type == 'regression':
            df = self.data_loader.get_all()
            if len(self.models) > 1:
                df[f'offset_{self.models[0]}'] = df[f'yPred_{self.models[0]}'] - df['yTrue']
                df[f'offset_{self.models[1]}'] = df[f'yPred_{self.models[1]}'] - df['yTrue']
            elif len(self.models) == 1:
                df[f'offset_{self.models[0]}'] = df[f'yPred_{self.models[0]}'] - df['yTrue']
            return df

        elif 'classification' in self.analysis_type:
            # extract list of class labels
            df_tmp = self.data_loader.get_all()[0]
            label_start_idx = list(df_tmp.columns).index('yTrue') + 1
            label_stop_idx = list(df_tmp.columns).index('model')
            class_labels = list(df_tmp.columns)[label_start_idx:label_stop_idx]

            ls_dfs_viz = []
            ls_dfs_by_label = []
            ls_dfs_by_label_state = []
            for df in self.data_loader.get_all():
                # extract df with info useful for miss-prediction viz                
                col_start_idx_to_extract = list(df.columns).index('yTrue')
                df_viz = df[list(df.columns)[col_start_idx_to_extract:]]
                pred_state_condition = df_viz['yPred-label'].astype('str') == df_viz['yTrue'].astype('str')
                df_viz['pred_state'] = np.where(pred_state_condition, 'correct', 'miss-predict')
                ls_dfs_viz.append(df_viz)

                # tapout df that is specific to each label-class
                dfs_specific_label = []
                dfs_state = []
                for label in class_labels:
                    df_label = df_viz[df_viz['yTrue'] == int(label)]
                    df_label = df_label[['yTrue', label, 'model', 'yPred-label', 'pred_state']]
                    dfs_specific_label.append(df_label)

                    is_exist_correct = True if len(df_label[df_label['pred_state'] == 'correct']) != 0 else False
                    is_exist_misspred = True if len(df_label[df_label['pred_state'] == 'miss-predict']) != 0 else False
                    state_dict = {}
                    state_dict['sample_size: '] = len(df_label)
                    state_dict['correct: '] = len(df_label[df_label['pred_state'] == 'correct']) if is_exist_correct else 0
                    state_dict['miss-predict:'] = len(df_label[df_label['pred_state'] == 'miss-predict']) if is_exist_misspred else 0
                    state_dict['accuracy: '] = round(state_dict['correct: '] / len(df_label), 4)
                    df_state = pd.DataFrame(state_dict, index=[0],).transpose().reset_index().rename(columns={0: 'state_value'})
                    dfs_state.append(df_state)

                ls_dfs_by_label.append(dfs_specific_label)
                ls_dfs_by_label_state.append(dfs_state)
            return ls_dfs_viz, ls_dfs_by_label, ls_dfs_by_label_state
