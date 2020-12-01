# Source: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

import pandas as pd
import gensim
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import PorterStemmer
from nltk.stem.porter import *
np.random.seed(2018)
from gensim import corpora, models

import nltk
nltk.download('wordnet')

#%% Load dataset
data = pd.read_csv(r"C:\Users\vinee\OneDrive - Massachusetts Institute of Technology\MIT\Fall 2020\6.867\Project\amzn_title_embs_per_art.csv",usecols = ["title","text"])
data['index'] = data.index