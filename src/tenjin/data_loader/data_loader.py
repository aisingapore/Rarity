import pandas as pd


class CSVDataLoader:
    def __init__(self, xFeatures_file, yTrue_file, yPred_file_list=[], model_names_list=[]):
        self.xFeatures = xFeatures_file
        self.yTrue = yTrue_file
        self.yPreds = yPred_file_list
        self.models = model_names_list
        assert len(self.yPreds)==len(self.models), 'no. of yPred_files must be equal to no. of model_names in correct order'

    def get_features(self):
        return pd.read_csv(self.xFeatures)
    
    def get_yTrue(self):
        yTrue = pd.read_csv(self.yTrue)
        yTrue.rename(columns={yTrue.columns[0]: 'yTrue'}, inplace=True)
        return yTrue
        
    def get_yPreds(self):
        yPred_ls = [pd.read_csv(yPred) for yPred in self.yPreds]
        for i in range(len(self.models)):
            yPred_ls[i]['model'] = self.models[i]
            yPred_ls[i]['yPred-label'] = yPred_ls[i][list(yPred_ls[i].columns)[:-1]].idxmax(axis=1)
        return yPred_ls
    
    def get_model_list(self):
        return self.models
    
    def get_all(self):
        df_ls = []
        for i in range(len(self.models)):
            df = pd.concat([self.get_features(), self.get_yTrue(), self.get_yPreds()[i]], axis=1)
            df_ls.append(df)
#         dfs = pd.concat(df_ls) # can remove this dfs if memory issue for big files
        return df_ls


class DataframeLoader:
    def __init__(self, df_xFeatures, df_yTrue, df_yPred_list=[], model_names_list=[]):
        self.xFeatures = df_xFeatures
        self.yTrue = df_yTrue.copy()
        self.yPreds = [df.copy() for df in df_yPred_list]
        self.models = model_names_list
        assert len(self.yPreds)==len(self.models), 'no. of yPred_files must be equal to no. of model_names in correct order'

    def get_features(self):
        return self.xFeatures

    def get_yTrue(self):
        self.yTrue.rename(columns={self.yTrue.columns[0]: 'yTrue'}, inplace=True)
        return self.yTrue

    def get_yPreds(self):
        for i in range(len(self.models)):
            self.yPreds[i]['model'] = self.models[i]
            self.yPreds[i]['yPred-label'] = self.yPreds[i][list(self.yPreds[i].columns)[:-1]].idxmax(axis=1)
        return self.yPreds

    def get_model_list(self):
        return self.models
        
    def get_all(self):
        df_ls = []
        for i in range(len(self.models)):
            df = pd.concat([self.xFeatures, self.get_yTrue(), self.get_yPreds()[i]], axis=1)
            df_ls.append(df)
#         dfs = pd.concat(df_ls) # can remove this dfs if memory issue for big files
        return df_ls