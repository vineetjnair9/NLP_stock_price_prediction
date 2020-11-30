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
rounding_digits = 10
regularized = False

number_iterations = 1000

refit = True
#------------------------------------## GET RELEVANT NUMERIC DATA ##------------------------------------#
def main(type_of_data_arg, ticker, home_directory):
    print("\n")
    print("RUNNING LINEAR REGRESSION on: " + type_of_data_arg + "...\n")


    
    X = pd.read_csv(home_directory + "/DataCSVs/" + type_of_data_arg + "_training_data_" + ticker + "_9_30_2012_9_30_2020.csv", index_col = 0) 
   
    # print("Entire X DF:")
    # print(X)


    #------------------------------------## PLOT CORRELATION MATRIX FOR ALL DATA ##------------------------------------#
    if len(X.columns) < 20:
        correlation_plot(X, "ALL_DATA", home_directory, ticker)

    #------------------------------------## CALCULATE VARIANCE INFLATION FACTORS ##------------------------------------#
    # print("Printing VIF for ALL Features:")
    # print(vif(X))
    # print("\n")



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
    if len(X.columns) < 20:
        correlation_plot(X, "ACTUAL_DATA", home_directory, ticker)

    #------------------------------------## CALCULATE VARIANCE INFLATION FACTORS ##------------------------------------#
    # print("Printing VIF for TRUE DF:")
    # print(vif(X))
    # print("\n")

    training_adj_r_squared = []
    testing_adj_r_squared = []
    training_mse_list = []
    testing_mse_list = []
    for i in range(number_iterations):


        ### TEST TRAIN SPLIT ###
        # print("Doing a test/train split.\n")
        train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.20)# random_state=42)
        # train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.10)#, random_state=42)



        #------------------------------------## FIT OLS ON TRAINING DF ###------------------------------------###
        ols_model = sm.OLS(train_Y, train_X)

        if regularized == False:
            model_fit = ols_model.fit()
            # print(model_fit.summary())
            ### Refit the model using only significant variables ###

            if refit == True:
                model_fit_df = results_summary_to_dataframe(model_fit)
                model_fit_df = model_fit_df[model_fit_df['pvals'] <= max(min(model_fit_df['pvals']), 0.1)] 
                train_X_new = train_X[model_fit_df.index.tolist()]
                ols_model = sm.OLS(train_Y, train_X_new)
                model_fit = ols_model.fit()
                train_fitted_Y = model_fit.predict(train_X_new)
                train_model_residuals = model_fit.resid            
                # print(model_fit.summary())

        elif regularized == True:
            model_fit = ols_model.fit_regularized(L1_wt=0.1, refit= False)
            print(model_fit.summary()) # NOT IMPLEMENTED

            # val_fitted_Y = model_fit.predict(val_X)
            # validation_mse = mse(val_Y, val_fitted_Y, rounding_digits)
            # hyperparameters_results.append(validation_mse)


            # # print("\n\n\n")
            # # print(hyperparameters_results)
            # plot_regularization_hyperparameters(l1_wts_to_test, hyperparameters_results, home_directory, ticker)



        #------------------------------------## CALCULATE FIT AND PLOT RESULTS ###------------------------------------###
        
        training_r_squared = model_fit.rsquared_adj
        # training_r_squared = model_fit.rsquared
        training_mse = mse(train_Y, train_fitted_Y, rounding_digits)

        training_adj_r_squared.append(training_r_squared)
        training_mse_list.append(training_mse)
    


        linreg_Plots(train_Y, train_fitted_Y, train_model_residuals, "TRAIN_" + type_of_data_arg, home_directory, ticker)



        #------------------------------------## TESTING MODEL ##------------------------------------#
        if refit == True:
            test_X = test_X[model_fit_df.index.tolist()]
        fitted_test_Y = model_fit.predict(test_X)
        resid_test_Y = np.array([test_Y[i] - fitted_test_Y[i] for i in range(len(test_Y))])


        #------------------------------------## CALCULATE FIT AND PLOT RESULTS ###------------------------------------###
        testing_rsquared = test_Y.corr(fitted_test_Y)**2
        testing_mse = mse(test_Y, fitted_test_Y, rounding_digits)

        testing_adj_r_squared.append(testing_rsquared)
        testing_mse_list.append(testing_mse)



    avg_train_r = statistics.mean(training_adj_r_squared)
    avg_test_r = statistics.mean(testing_adj_r_squared)
    avg_train_mse = statistics.mean(training_mse_list)
    avg_test_mse = statistics.mean(testing_mse_list)
    print("Average Adjusted Training R^2: " + str(avg_train_r))
    print("Average Testing R^2: " + str(avg_test_r)) 
    print("Average Training MSE: " + str(avg_train_mse))
    print("Average Testing MSE: " + str(avg_test_mse))

    plot_metrics_for_many_iterations(training_adj_r_squared, training_mse_list, testing_mse_list, testing_adj_r_squared,type_of_data_arg, home_directory, ticker)
    linreg_Plots(test_Y, fitted_test_Y, resid_test_Y, "TEST_" + type_of_data_arg, home_directory, ticker)

    # return([ticker, avg_train_r, avg_train_mse, avg_test_r, avg_test_mse])

    return([avg_train_r, avg_test_r, avg_train_mse,  avg_test_mse])

if __name__ == "__main__": 
    tickers = sys.argv[1].split(",")
#     to_write_numeric_only = [["ticker", 'training R Squared', 'Training MSE', 'Testing R Squared', 'Testing MSE']]
#     to_write_numeric_and_text_only = [["Ticker", 'Training R Squared', 'Training MSE', 'Testing R Squared', 'Testing MSE']]

#     for ticker in tickers:
#         to_write_numeric_only.append(main("numeric", ticker, home_directory))
#         to_write_numeric_and_text_only.append(main("numeric_and_text", ticker, home_directory))       # UNCOMMENT FOR TEXT DATA TOO


#     with open(home_directory + "/LinReg_Results/" + "numeric.csv", "w") as fileout:
#         csvobj = csv.writer(fileout)
#         csvobj.writerows(to_write_numeric_only)
    
# # UNCOMMENT FOR TEXT DATA TOO
#     with open(home_directory + "/LinReg_Results/" + "numeric_and_text.csv", "w") as fileout:
#         csvobj = csv.writer(fileout)
#         csvobj.writerows(to_write_numeric_and_text_only)

#         print("\n\n")
    to_write_results = [['Ticker', 'Percent Change in Training Adjusted R2', 'Percent Change in Testing R2', 'Percent Change in Training MSE', 'Percent Change in Testing MSE']]


    for ticker in tickers:
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

    os.system("python3 run_MakeLatexTables.py")
