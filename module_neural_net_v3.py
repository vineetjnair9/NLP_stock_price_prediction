import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout, Bidirectional, Input, concatenate, add, multiply, GRU, SimpleRNN
from keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, GlobalMaxPooling1D, Lambda
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVR

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt