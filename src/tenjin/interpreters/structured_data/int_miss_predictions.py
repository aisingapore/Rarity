import pandas as pd

from tenjin.interpreters.structured_data.base_interpreters import BaseInterpreters


class IntMissPredictions(BaseInterpreters):
    """
    - Transform raw data into input format suitable for plotting with plotly

    Arguments:
        data_loader {class object} -- class object from data_loader pipeline

    Returns:
        if analysis_type == 'regression'
            df {dataframe} -- dataframe storing info needed for visualizer
        elif analysis_type == 'classification'
            ls_dfs_viz {list} -- list of dataframes for overview
            ls_class_labels {list} -- list of class labels
            ls_dfs_by_label {list} -- list of dataframes by individual label class
            ls_dfs_by_label_state {list} -- list of dataframes storing basic stats of each label class
        """
    def __init__(self, data_loader):
        super().__init__(data_loader)

    def xform(self):
        if self.analysis_type == 'regression':
            df = super().get_df_with_offset_values()
            return df

        elif 'classification' in self.analysis_type:
            ls_dfs_viz, ls_class_labels = super().get_df_with_probability_values()

            ls_dfs_by_label = []
            ls_dfs_by_label_state = []
            for df_viz in ls_dfs_viz:
                # tapout df that is specific to each label-class
                dfs_specific_label = []
                dfs_state = []
                for label in ls_class_labels:
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
            return ls_dfs_viz, ls_class_labels, ls_dfs_by_label, ls_dfs_by_label_state
