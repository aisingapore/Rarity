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
