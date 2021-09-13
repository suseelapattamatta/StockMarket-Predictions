import pandas as pd
import numpy as np


def evaluation(y_true,y_pred,name="Prediction"):
    tp = len( y_true[ (y_true==1) & (y_pred==1) ])
    fp = len( y_true[ (y_true==0) & (y_pred==1) ])
    tn = len( y_true[ (y_true==0) & (y_pred==0) ])
    fn = len( y_true[ (y_true==1) & (y_pred==0) ])
    
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f_measure = (2 * precision * recall) / (precision + recall)
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    
    return pd.Series([precision, recall, f_measure, accuracy],
                     index=["precision","recall","f_measure","accuracy"],
                     name = name)


def dummy_previous(y_true):
    """Simple model that predicts the previous value"""
    return evaluation(y_true[1:], y_true[:-1],name="Naive Previous")

def dummy_increase(y_true):
    """Simple model that always predicts one (increase)"""
    return evaluation( y_true, np.ones(y_true.shape), name="Naive Increase" )


def reg_to_binary(s):
    """Function to turn a time series into binary classification labels"""
    return (np.diff(s,n=1)>0).astype(int)