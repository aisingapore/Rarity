import numpy as np


class BaseInterpreters:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.analysis_type = data_loader.get_analysis_type()
        self.models = data_loader.get_model_list()

    def get_df_with_offset_values(self):
        '''
        For use in regression task only
        '''
        df = self.data_loader.get_all()
        df[f'offset_{self.models[0]}'] = df[f'yPred_{self.models[0]}'] - df['yTrue']
        if len(self.models) > 1:
            df[f'offset_{self.models[1]}'] = df[f'yPred_{self.models[1]}'] - df['yTrue']            
        return df

    def get_df_with_probability_values(self):
        '''
        For use in classification task only
        '''
        # extract list of class labels
        df_tmp = self.data_loader.get_all()[0]
        label_start_idx = list(df_tmp.columns).index('yTrue') + 1
        label_stop_idx = list(df_tmp.columns).index('model')
        ls_class_labels = list(df_tmp.columns)[label_start_idx:label_stop_idx]

        ls_dfs_viz = []
        for df in self.data_loader.get_all():
            # extract df with info useful for miss-prediction viz
            col_start_idx_to_extract = list(df.columns).index('yTrue')
            df_viz = df.loc[:, list(df.columns)[col_start_idx_to_extract:]]
            pred_state_condition = df_viz['yPred-label'].astype('str') == df_viz['yTrue'].astype('str')
            df_viz['pred_state'] = np.where(pred_state_condition, 'correct', 'miss-predict')

            # re-arrange the columns orders for viz need at features-module
            org_cols = list(df_viz.columns)
            org_cols.remove('model')
            new_position_yTrue_col_idx = org_cols.index('yPred-label')
            new_cols_order = ['model'] + org_cols[1:new_position_yTrue_col_idx] + ['yTrue'] + org_cols[new_position_yTrue_col_idx:]
            df_viz_final = df_viz.loc[:, new_cols_order]
            ls_dfs_viz.append(df_viz_final)
        return ls_dfs_viz, ls_class_labels