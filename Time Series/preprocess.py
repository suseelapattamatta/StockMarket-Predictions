import sklearn as sk
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
import string
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import tensorflow
from sklearn.model_selection import train_test_split
from collections import Counter
from pandas import DataFrame
from matplotlib import pyplot
import re
from nltk.stem import WordNetLemmatizer 
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler



def process_news(news):
    _news = news.replace('b\"', "")
    _news = _news.replace('b\'', "")
    _news = _news.lower()
    _news = re.sub("[^a-zA-Z]", " ",_news)
    _news = re.sub('[\s]+', ' ', _news)
    
    tokens = _news.split(" ")
    if "" in tokens:
        tokens.remove("")
    
    lemmatizer = WordNetLemmatizer() 
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    #remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    
    _news = ' '.join(tokens)    
     
    return _news


def read_data():

    data = pd.read_csv("Combined_News_DJIA.csv")
    
    dfs = []
    data["News"] = ""
    for i in range(1,25):
        col = "Top"+str(i)
        data["News"] = data["News"] +" "+ data[col]
    data = data.dropna()
    data['PreProcessedNews'] = data['News'].map(process_news)
    
    data = data[['Date', 'News', 'PreProcessedNews', 'Label']]
    
    stock_prices = "upload_DJIA_table.csv"
    stock_data = pd.read_csv(stock_prices)
    
    print(data.head(2))
    print(stock_data.head(2))
    
    merged_dataframe = pd.merge(data, stock_data, how='inner', on = 'Date')

    
    
    
    Xy_train = merged_dataframe[:int(len(data)*0.6)]
    Xy_valid = merged_dataframe[int(len(data)*0.6):int(len(data)*0.8)]
    Xy_test = merged_dataframe[int(len(data)*0.8):]
    
    return merged_dataframe, Xy_train, Xy_valid, Xy_test



def create_timeShiftSet(X_train,X_test,y_train,y_test,shift=1,del_first=False, scaleX=False, scaleY=False):
    """Function to build a training and testing set composed of previous time steps
    It returns new X_train and X_test matrixes."""
    
    x_gather = pd.concat((X_train,X_test))
    y_gather = pd.concat((y_train,y_test))
    
    # Create the shifted columns
    if type(shift)==int: # if a number of shifts is given
        newX = pd.concat([x_gather.shift(k) for k in range(1+del_first.real,1+del_first.real+shift)],axis=1)
        newX.columns = range(shift)
        newX = newX.interpolate(method="linear").iloc[1+shift:] # Shift creates NaN...
        
        newX_train = newX.loc[X_train.index[1+shift:],:].values
        newX_test = newX.loc[X_test.index,:].values
        newy_train = y_gather.loc[X_train.index[1+shift:]].values
        newy_test = y_gather.loc[X_test.index].values
        
    elif type(shift) in [list,tuple]: # if a list of shifts to respect is given
        newX = pd.concat([x_gather.shift(k) for k in shift],axis=1)
        newX.columns = shift
        newX = newX.interpolate(method="linear").iloc[max(shift):del_first.real:] # Shift creates NaN...
    
        newX_train = newX.loc[X_train.index[max(shift):],:].values
        newX_test = newX.loc[X_test.index,:].values
        newy_train = y_gather.loc[X_train.index[max(shift):]].values
        newy_test = y_gather.loc[X_test.index].values
    
    if scaleX:
        xScaler = StandardScaler().fit(newX_train)
        newX_train = xScaler.transform(newX_train)
        newX_test = xScaler.transform(newX_test)
    if scaleY:
        yScaler = StandardScaler().fit(newy_train.reshape(-1,1))
        newy_train = yScaler.transform(newy_train.reshape(-1,1)).ravel()
        newy_test = yScaler.transform(newy_test.reshape(-1,1)).ravel()
    
    return newX_train, newX_test, newy_train, newy_test