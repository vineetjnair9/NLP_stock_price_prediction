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

# ticker = 'AMZN'
ticker_vec = ["AMZN", "GE", "T", "PFE", "GS"]
rounding_digits = 10
regularized = False

#------------------------------------## GET RELEVANT NUMERIC DATA ##------------------------------------#
def main(type_of_data_arg, ticker, home_directory):
    print("\n")
    print("RUNNING LINEAR REGRESSION on: " + type_of_data_arg + "...\n")


    
    X = pd.read_csv(home_directory + "/DataCSVs/" + type_of_data_arg + "_training_data_" + ticker + "_9_30_2012_9_30_2020.csv", index_col = 0) 
   
    print("Entire X DF:")
    print(X)


    #------------------------------------## PLOT CORRELATION MATRIX FOR ALL DATA ##------------------------------------#
    correlation_plot(X, "ALL_DATA", home_directory, ticker)

    #------------------------------------## CALCULATE VARIANCE INFLATION FACTORS ##------------------------------------#
    print("Printing VIF for ALL Features:")
    print(vif(X))
    print("\n")



    #------------------------------------## PREPARE MATRIX FOR REGRESSION AND TEST-TRAIN SPLIT ##------------------------------------#
    y = X['TARGET'].copy()
    X.drop(['TARGET'], inplace = True, axis = 1)

    ####### KEEP ONLY CERTAIN COLUMNS OF DF #######
    # column_names_to_keep = ['STOCK_PRICE_Close', 'VIX_High','STOCK_PRICE_Volume']
    # column_names_to_keep.extend([i for i in X.columns if 'textual_feature' in i])
    # X = X[column_names_to_keep].copy() 
    X.drop(['DOW_Open', 'NASDAQ_Open', 'SP_Open'], axis = 1, inplace = True)
    X = sm.add_constant(X)


    #------------------------------------## PLOT CORRELATION MATRIX FOR TRUE DATA ##------------------------------------#
    correlation_plot(X, "ACTUAL_DATA", home_directory, ticker)

    #------------------------------------## CALCULATE VARIANCE INFLATION FACTORS ##------------------------------------#
    print("Printing VIF for TRUE DF:")
    print(vif(X))
    print("\n")

    ### TEST TRAIN SPLIT ###
    print("Doing a test/train split.\n")
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.20)#, random_state=42)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.10)#, random_state=42)



    #------------------------------------## FIT OLS ON TRAINING DF ###------------------------------------###
    ols_model = sm.OLS(train_Y, train_X)

    if regularized == False:
        model_fit = ols_model.fit()
        print(model_fit.summary())
        train_fitted_Y = model_fit.predict(train_X)
        train_model_residuals = model_fit.resid

    elif regularized == True:
        hyperparameters_results = []
        l1_wts_to_test = np.linspace(.0001, .999, 20)
        for i in l1_wts_to_test:
            print("i is: " + str(i))
            model_fit = ols_model.fit_regularized(L1_wt=i, refit= False)
            val_fitted_Y = model_fit.predict(val_X)
            validation_mse = mse(val_Y, val_fitted_Y, rounding_digits)
            hyperparameters_results.append(validation_mse)
            # train_fitted_Y = model_fit.predict(train_X)
            # train_mse = mse(train_Y, train_fitted_Y, rounding_digits)
            # hyperparameters_results.append(train_mse)


        print("\n\n\n")
        print(hyperparameters_results)
        plot_regularization_hyperparameters(l1_wts_to_test, hyperparameters_results, home_directory, ticker)



    print("Printing TRAINING RESULTS: \n")

    #------------------------------------## CALCULATE FIT AND PLOT RESULTS ###------------------------------------###
    training_r_squared = model_fit.rsquared
    print("TRAINING R-Squared: " + str(round(training_r_squared, rounding_digits)))
    # training_sse = sse(train_Y, train_fitted_Y, rounding_digits)
    # print("TRAINING SSE: " + str(training_sse))
    training_mse = mse(train_Y, train_fitted_Y, rounding_digits)
    print("TRAINING MSE: " + str(training_mse))


    print("\n")

    linreg_Plots(train_Y, train_fitted_Y, train_model_residuals, "TRAIN_" + type_of_data_arg, home_directory, ticker)



    #------------------------------------## TESTING MODEL ##------------------------------------#
    fitted_test_Y = model_fit.predict(test_X)
    resid_test_Y = np.array([test_Y[i] - fitted_test_Y[i] for i in range(len(test_Y))])


    #------------------------------------## CALCULATE FIT AND PLOT RESULTS ###------------------------------------###
    testing_rsquared = test_Y.corr(fitted_test_Y)**2
    print("TESTING R-Squared: " + str(round(testing_rsquared, rounding_digits)))
    # testing_sse = sse(test_Y, fitted_test_Y, rounding_digits)
    # print("TESTING SSE: " + str(testing_sse))
    testing_mse = mse(test_Y, fitted_test_Y, rounding_digits)
    print("TESTING MSE: " + str(testing_mse))


    linreg_Plots(test_Y, fitted_test_Y, resid_test_Y, "TEST_" + type_of_data_arg, home_directory, ticker)

    return([ticker, training_r_squared, training_mse, testing_rsquared, testing_mse])

if __name__ == "__main__": 
    to_write_numeric_only = [["ticker", 'training R Squared', 'Training MSE', 'Testing R Squared', 'Testing MSE']]
    to_write_numeric_and_text_only = [["Ticker", 'Training R Squared', 'Training MSE', 'Testing R Squared', 'Testing MSE']]

    for ticker in ticker_vec:
        to_write_numeric_only.append(main("numeric", ticker, home_directory))
        # to_write_numeric_and_text_only.append(main("numeric_and_text", ticker))       # UNCOMMENT FOR TEXT DATA TOO


    with open(home_directory + "/LinReg_Results/" + "numeric.csv", "w") as fileout:
        csvobj = csv.writer(fileout)
        csvobj.writerows(to_write_numeric_only)
    pp.pprint(to_write_numeric_only)
    
# UNCOMMENT FOR TEXT DATA TOO
    # with open(home_directory + "/LinReg_Results/" + "numeric_and_text.csv", "w") as fileout:
    #     csvobj = csv.writer(fileout)
    #     csvobj.writerows(to_write_numeric_and_text_only)
    # pp.pprint(to_write_numeric_and_text_only)




