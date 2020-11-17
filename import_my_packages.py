import csv
import sys
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import pprint as pp
import random
from scipy.stats import norm
import sklearn
import scipy.spatial.distance as sdist
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import resample
from sklearn import metrics
import scipy
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
import seaborn
import seaborn as sns
# from gurobipy import Model,GRB
import os
import datetime
import calendar
import statistics
import statsmodels.api as sm
from sklearn import linear_model as lm
import warnings
from operator import itemgetter
import pandasql
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context # accept all URL access
warnings.filterwarnings("ignore", category=RuntimeWarning)
import statistics
from pandas.plotting import register_matplotlib_converters
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from lmfit.models import StepModel, LinearModel
from striprtf.striprtf import rtf_to_text
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import re
from compose import compose
import scipy.fftpack
from datetime import datetime
from statsmodels.nonparametric.kernel_regression import KernelReg
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
register_matplotlib_converters()
from scipy.optimize import fsolve
from sympy import *
from scipy.optimize import root
pd.options.mode.chained_assignment = None
from matplotlib.ticker import MaxNLocator
import stargazer
import yfinance as yf
import quandl
import urllib.request as ur
from bs4 import BeautifulSoup
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

import nltk
import string
from string import digits
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
