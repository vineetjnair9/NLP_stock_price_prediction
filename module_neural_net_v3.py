import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Input
from keras import backend as K
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout, Bidirectional, Input, concatenate, add, multiply, GRU, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, GlobalMaxPooling1D, Lambda
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import os

#%%
def get_inputs(ticker, onehot):

  inputs_no_text = pd.read_csv(f'numeric_training_data_{ticker}_9_30_2012_9_30_2020.csv',index_col=0)
  inputs_with_text = pd.read_csv(f'numeric_and_text_training_data_{ticker}_9_30_2012_9_30_2020_text.csv',index_col=0)
  inputs_with_title = pd.read_csv(f'numeric_and_text_training_data_{ticker}_9_30_2012_9_30_2020_title.csv',index_col=0)
  inputs_parsed_titles = pd.read_csv(f'numeric_and_text_training_data_{ticker}_9_30_2012_9_30_2020_PARSED_ARTICLES_TITLE.csv',index_col=0)
  inputs_parsed_text = pd.read_csv(f'numeric_and_text_training_data_{ticker}_9_30_2012_9_30_2020_PARSED_ARTICLES_TEXT.csv',index_col=0)

  if (ticker == 'ALL_TICKERS' and not onehot):
    inputs_no_text = inputs_no_text.drop(columns=['ticker_AMZN','ticker_GS','ticker_PFE','ticker_SIEGY','ticker_TSLA','ticker_UL'])
    inputs_with_text = inputs_with_text.drop(columns=['ticker_AMZN','ticker_GS','ticker_PFE','ticker_SIEGY','ticker_TSLA','ticker_UL'])
    inputs_with_title = inputs_with_title.drop(columns=['ticker_AMZN','ticker_GS','ticker_PFE','ticker_SIEGY','ticker_TSLA','ticker_UL'])
    inputs_parsed_titles = inputs_parsed_titles.drop(columns=['ticker_AMZN','ticker_GS','ticker_PFE','ticker_SIEGY','ticker_TSLA','ticker_UL'])
    inputs_parsed_text = inputs_parsed_text.drop(columns=['ticker_AMZN','ticker_GS','ticker_PFE','ticker_SIEGY','ticker_TSLA','ticker_UL'])

  l = [inputs_with_text, inputs_with_title, inputs_parsed_titles, inputs_parsed_text]
  res = []
  for df in l:
      column_order = ["TARGET","STOCK_PRICE_Open","VIX_Open",	"NASDAQ_Open",	"DOW_Open",	"SP_Open",	"Mkt-RF",	"SMB", "HML", "RMW", "CMA", "RF"]
      column_order.extend([str(i) for i in range(0, 300)])
      df = df[column_order]
      res.append(df)

  return inputs_no_text, res[0], res[1], res[2], res[3]

#%%
def split_train_val_test(features, train=0.8, val=0.1):
    shuffled = np.random.RandomState(0).permutation(features.index)
    n_train = int(len(shuffled) * train)
    n_val = int(len(shuffled) * val)
    i_train, i_val, i_test = shuffled[:n_train], shuffled[n_train: n_train + n_val], shuffled[-n_val:]
    return features.loc[i_train], features.loc[i_val], features.loc[i_test] # Confirm that (x,y) data points organized by rows

#%%
def split_train_val_test_seq(features, train=0.8, val=0.1): # Sequential time series data for RNNs
    rows = len(features)
    num_train = int(np.floor(rows*train))
    num_val = int(np.floor(rows*val))
    return features.iloc[:num_train], features.iloc[num_train+1:num_train+num_val+1], features.iloc[num_train+num_val+1:] 

#%%
def one_hot(y):
  # encode class values as integers
  encoder = LabelEncoder()
  encoder.fit(y)
  encoded_Y = encoder.transform(y)
  # convert integers to dummy variables (i.e. one hot encoded)
  dummy_Y = np_utils.to_categorical(encoded_Y)  
  return dummy_Y

#%%
def run_network(ticker,data,nn_type,sequential,onehot):

  inputs_no_text, inputs_with_text, inputs_with_title, inputs_parsed_titles, inputs_parsed_text = get_inputs(ticker,onehot=True)

  if data == 'numerical':
    inputs = inputs_no_text
  elif data == 'text':
    inputs = inputs_with_text
  elif data == 'titles':
    inputs = inputs_with_title
  elif data == 'parsed titles':
    inputs = inputs_parsed_titles
  else:
    inputs = inputs_parsed_text

  if (sequential):
      train, val, test = split_train_val_test_seq(inputs, train=0.8, val=0.1)
  else:
      train, val, test = split_train_val_test(inputs, train=0.8, val=0.1)
      
  X_train = train[train.columns.difference(['TARGET'])]
  y_train = train['TARGET']

  X_val = val[val.columns.difference(['TARGET'])]
  y_val = val['TARGET']

  X_test = test[test.columns.difference(['TARGET'])]
  y_test = test['TARGET']

  y_train = one_hot(y_train)
  y_val = one_hot(y_val)
  y_test = one_hot(y_test)

  if (not onehot):
    X_train = (X_train - X_train.mean())/X_train.std()
    X_val = (X_val - X_val.mean())/X_val.std()
    X_test = (X_test - X_test.mean())/X_test.std()
  else:
    for i in X_train.columns:
      if i not in ['ticker_AMZN','ticker_GS','ticker_PFE','ticker_SIEGY','ticker_TSLA','ticker_UL']:
        X_train[i] = (X_train[i] - X_train[i].mean())/X_train[i].std()
        X_val[i] = (X_val[i] - X_val[i].mean())/X_val[i].std()
        X_test[i] = (X_test[i] - X_test[i].mean())/X_test[i].std()

  history, scores = nn_type(X_train, y_train, X_val, y_val, X_test, y_test)

  plt.plot(history.history['accuracy'],label = 'Training')
  plt.plot(history.history['val_accuracy'],label='Validation')
  plt.title('Prediction accuracy using Neural Net')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend()
  plt.tight_layout() 
  
  print('Test set loss: ', round(scores[0],4))
  print('Test set accuracy: ', round(scores[1],4))
