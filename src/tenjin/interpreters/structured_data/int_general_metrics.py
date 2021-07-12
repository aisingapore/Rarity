"""
Version : 
---------
draft 3.0 [ 5 July 2021 - dash ]

"""


class GeneralMetrics:
    """
    - Transform raw data into input format suitable for plotting with plotly
    - General metrics cover confusion matrix, classification report, roc curve and precisionRecall curve

    Arguments:
        data_loader {class object} -- class object from data_loader pipeline
        viz_plot {str} -- visualization type : 'confMat', 'classRpt', 'rocAUC' or 'preRacall' 

    Returns:
        yTrue {pd.series} -- actual labels
        yPred {list of list} -- Predicted labels of different models
        model_names (list) -- name of models used to produce yPred
        """
    def __init__(self, data_loader, viz_plot=None):
        self.data_loader = data_loader
        self.viz_plot = viz_plot

    def xform(self):
        model_names = self.data_loader.get_model_list()
        yTrue = self.data_loader.get_yTrue()
        yTrue = yTrue['yTrue'].astype('string')
        
        preds = self.data_loader.get_yPreds()
        if self.viz_plot in ['confMat', 'classRpt']:
            yPred = [pred['yPred-label'] for pred in preds]
        elif self.viz_plot in ['rocAuc', 'precRecall']:
            is_multiclass = len(set(yTrue)) > 2
            if is_multiclass:
                yPred = []
                for pred in preds:
                    label_keys = pred['yPred-label']
                    pred_values = pred[pred.columns[:-2]].max(axis=1)
                    pred_tmp = [(label_keys[i], pred_values[i]) for i in range(len(label_keys))]
                    yPred.append(pred_tmp)
            else:
                yPred = [pred[pred.columns[-3]] for pred in preds]
            
        return yTrue, yPred, model_names


