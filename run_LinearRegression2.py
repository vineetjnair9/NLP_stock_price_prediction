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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

##################### PLOT SETTINGS #####################
font_dict = {'size' : 40, 'family': 'serif'}
font_dict_legend = {'size' : 20, 'family': 'serif'}
tick_size = 30
###################################################################

home_directory = os.getcwd()

#------------------------------------## SPECIFY SETTINGS ##------------------------------------#

# ticker = 'AMZN'
ticker_vec = ["AMZN", "GE", "T", "PFE", "GS"]
ticker_vec = ["AMZN"]

rounding_digits = 10
regularized = False

cv_splits = 10
alpha = 0.001 #higher = better training fit
#------------------------------------## GET RELEVANT NUMERIC DATA ##------------------------------------#
def main(type_of_data_arg, ticker, home_directory):
    print("\n")
    print("RUNNING LINEAR REGRESSION on: " + type_of_data_arg + "...\n")


    X = pd.read_csv(home_directory + "/DataCSVs/" + type_of_data_arg + "_training_data_" + ticker + "_9_30_2012_9_30_2020.csv", index_col = 0) 

    #------------------------------------## PREPARE MATRIX FOR REGRESSION AND TEST-TRAIN SPLIT ##------------------------------------#
    y = X['TARGET'].copy()
    X.drop(['TARGET'], inplace = True, axis = 1)
    X.drop(['DOW_Open', 'NASDAQ_Open', 'SP_Open'], axis = 1, inplace = True)
    X = sm.add_constant(X)

    print(X)

    training_adj_r_squared = []
    testing_adj_r_squared = []
    training_mse_list = []
    testing_mse_list = []


    #------------------------------------## FIT OLS ON TRAINING DF ###------------------------------------###

    ols2 = Lasso(alpha=alpha, fit_intercept= True)
    # ols2 = Ridge(alpha=alpha, fit_intercept= True)

    ols2 = LinearRegression()

    cv_results = cross_validate(ols2, X, y, scoring=('r2', 'neg_mean_squared_error'), cv=cv_splits, return_train_score=True)
    training_adj_r_squared = cv_results['train_r2']
    testing_adj_r_squared = cv_results['test_r2']
    training_mse_list = cv_results['train_neg_mean_squared_error']
    testing_mse_list = cv_results['test_neg_mean_squared_error']

    plot_metrics_for_many_iterations(training_adj_r_squared, training_mse_list, testing_mse_list, testing_adj_r_squared, type_of_data_arg, home_directory, ticker)

    avg_train_r = statistics.mean(training_adj_r_squared)
    avg_test_r = statistics.mean(testing_adj_r_squared)
    avg_train_mse = statistics.mean(training_mse_list)
    avg_test_mse = statistics.mean(testing_mse_list)
    print("Average Training R^2: " + str(avg_train_r))
    print("Average Testing R^2: " + str(avg_test_r)) 
    print("Average Training Negative MSE: " + str(avg_train_mse))
    print("Average Testing Negative MSE: " + str(avg_test_mse))

    return(avg_train_r, avg_test_r, avg_train_mse, avg_test_mse)




if __name__ == "__main__": 

    to_write_results = [['Ticker', 'Percent Change in Training R2', 'Percent Change in Testing R2', 'Percent Change in Training MSE', 'Percent Change in Testing MSE']]
    for ticker in ticker_vec:
        numeric_avg_train_r, numeric_avg_test_r, numeric_avg_train_mse, numeric_avg_test_mse = main("numeric", ticker, home_directory)
        numeric_and_text_avg_train_r, numeric_and_text_avg_test_r, numeric_and_text_avg_train_mse, numeric_and_text_avg_test_mse = main("numeric_and_text", ticker, home_directory)
    
        print("\n\n")
        pc_train_r2 = percent_change(numeric_and_text_avg_train_r, numeric_avg_train_r, 1)
        print("Using text features results in " + str(pc_train_r2) + "% percent change in average training R-squared.")

        pc_test_r2 = percent_change(numeric_and_text_avg_test_r, numeric_avg_test_r, 1)
        print("Using text features results in " + str(pc_test_r2) + "% percent change in average testing R-squared.")

        pc_train_mse = percent_change(numeric_and_text_avg_train_mse, numeric_avg_train_mse)
        print("Using text features results in " + str(pc_train_mse) + "% percent change in average training mse.")

        pc_test_mse = percent_change(numeric_and_text_avg_test_mse, numeric_avg_test_mse)
        print("Using text features results in " + str(pc_test_mse) + "% percent change in average testing mse.")
        to_write_results.append([ticker, pc_train_r2, pc_test_r2, pc_train_mse, pc_test_mse])
    
    with open(home_directory + "/LinReg_Results/" + "linreg_cv_percent_changes.csv", "w") as fileout:
        csvobj = csv.writer(fileout)
        csvobj.writerows(to_write_results)

