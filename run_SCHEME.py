from module_scraper import *
from module_numericFinancialData import * 
from import_my_packages import * 
from module_featurizer import *
from module_linregHelper import * 

from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import spatial
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor 
##################### PLOT SETTINGS #####################
font_dict = {'size' : 40, 'family': 'serif'}
font_dict_legend = {'size' : 20, 'family': 'serif'}
tick_size = 30
###################################################################

home_directory = os.getcwd()

#------------------------------------## SPECIFY SETTINGS ##------------------------------------#
ticker_vec = ["AMZN", "GE", "T", "PFE", "GS"]
ticker_vec = ["AMZN"]



#------------------------------------## RUN ##------------------------------------#
ticker_vec = ','.join(map(str, ticker_vec)) 
# os.system("python3 run_MakeDataFrames.py" + " " + ticker_vec)
os.system("python3 run_kmeans_median.py" + " " + ticker_vec)
# os.system("python3 run_LinearRegression.py"+ " " +ticker_vec)
# os.system("python3 run_MakeLatexTables.py"+ " " + ticker_vec)

#------------------------------------## RUN MANUALLY ##------------------------------------#
# Rscript ARIMA.R