from module_scraper import *
from module_numericFinancialData import * 
from import_my_packages import * 
from module_featurizer import *
from module_linregHelper import * 

from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
# from sklearn_extra.cluster import KMedoids
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



#------------------------------------## SPECIFY SETTINGS ##------------------------------------#

ticker = 'AMZN'
start_date = datetime(2019, 10, 30)
start_date_string = str(start_date.month) + "/" + str(start_date.day) + "/" + str(start_date.year) 
end_date = datetime(2020, 10,30)
end_date_string = str(end_date.month) + "/" + str(end_date.day) + "/" + str(end_date.year) 


#------------------------------------## GET RELEVANT NUMERIC DATA ##------------------------------------#
def main(type_of_data_arg):
    print("\n")
    print("RUNNING LINEAR REGRESSION on: " + type_of_data_arg + "...\n")

    home_directory = os.getcwd()

    try:
        X = pd.read_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0) 
    except:
        print("DF requested does not yet exist. Writing new data now...")
        if type_of_data_arg == "numeric":
            training_numeric_df = create_numeric_training_data(ticker, start_date, end_date)
            training_numeric_df.to_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv") 
        
        elif type_of_data_arg == "numeric_and_text":

            ### Write numeric data ###
            try: # if file already exists
                training_numeric_df = pd.read_csv(home_directory + "/numeric_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0) 
            except:
                training_numeric_df = create_numeric_training_data(ticker, start_date, end_date)
                training_numeric_df.to_csv(home_directory + "/" + "numeric" + "_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv") 
                training_numeric_df = pd.read_csv(home_directory + "/numeric_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0) 

            ### Write text data ###
            try: # if file already exists
                training_text_df = pd.read_csv(home_directory + "/text_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0) 
            except:
###### NEED TO FIX TEXTUAL FEATURES HERE ####
                training_text_data = pd.DataFrame([['88888888', '99999999' ] for i in range(len(training_numeric_df.index))], columns = ['textual_feature_1', 'textual_feature_2']) # NEED TO SPECIFY SOME FUNCTION HERE
                training_text_data.index = training_numeric_df.index
###### NEED TO FIX TEXTUAL FEATURES HERE ####
                training_text_data.to_csv(home_directory + "/" + "text" + "_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv") 
                training_text_df = pd.read_csv(home_directory + "/text_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0) 

            ### Create joint df ###
            training_numeric_and_text_df = pd.merge(training_numeric_df, training_text_df, how='outer', left_index=True, right_index=True)
            training_numeric_and_text_df.to_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv") 

        X = pd.read_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0) 

    print("Entire X DF:")
    print(X)


    #------------------------------------## PLOT CORRELATION MATRIX FOR ALL DATA ##------------------------------------#
    correlation_plot(X, "ALL_DATA", home_directory)

    #------------------------------------## CALCULATE VARIANCE INFLATION FACTORS ##------------------------------------#
    print("Printing VIF for ALL Features:")
    print(vif(X))
    print("\n")



    #------------------------------------## PREPARE MATRIX FOR REGRESSION AND TEST-TRAIN SPLIT ##------------------------------------#
    y = X['TARGET'].copy()
    X.drop(['TARGET'], inplace = True, axis = 1)

    ####### KEEP ONLY CERTAIN COLUMNS OF DF #######
    column_names_to_keep = ['STOCK_PRICE_Close', 'VIX_High','STOCK_PRICE_Volume']
    column_names_to_keep.extend([i for i in X.columns if 'textual_feature' in i])
    X = X[column_names_to_keep].copy() 
    X = sm.add_constant(X)


    #------------------------------------## PLOT CORRELATION MATRIX FOR TRUE DATA ##------------------------------------#
    correlation_plot(X, "ACTUAL_DATA", home_directory)

    #------------------------------------## CALCULATE VARIANCE INFLATION FACTORS ##------------------------------------#
    print("Printing VIF for TRUE DF:")
    print(vif(X))
    print("\n")

    ### TEST TRAIN SPLIT ###
    print("Doing a test/train split.\n")
    train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.20)#, random_state=42)



    #------------------------------------## FIT OLS ON TRAINING DF ###------------------------------------###
    ols_model = sm.OLS(train_Y, train_X)
    model_fit = ols_model.fit()
    print("Printing TRAINING RESULTS: \n")
    print(model_fit.summary())
    train_fitted_Y = model_fit.predict(train_X)
    train_model_residuals = model_fit.resid



    #------------------------------------## CALCULATE FIT AND PLOT RESULTS ###------------------------------------###
    training_r_squared = model_fit.rsquared
    print("TRAINING R-Squared: " + str(round(training_r_squared, 5)))
    training_mse = model_fit.mse_resid / len(test_X)
    print("TRAINING Avg. MSE: " + str(round(training_mse, 5)))
    training_mape = mape(train_Y, train_fitted_Y)
    print("TRAINING MAPE (%): " + str(training_mape))
    print("\n")

    linreg_Plots(train_Y, train_fitted_Y, train_model_residuals, "TRAIN_" + type_of_data_arg, home_directory)



    #------------------------------------## TESTING MODEL ##------------------------------------#
    fitted_test_Y = model_fit.predict(test_X)
    resid_test_Y = np.array([test_Y[i] - fitted_test_Y[i] for i in range(len(test_Y))])


    #------------------------------------## CALCULATE FIT AND PLOT RESULTS ###------------------------------------###
    testing_rsquared = test_Y.corr(fitted_test_Y)**2
    print("TRAINING R-Squared: " + str(round(testing_rsquared, 5)))
    testing_mse = mean_squared_error(fitted_test_Y, test_Y) / len(test_X)
    print("TESTING Avg. MSE: " + str(round(testing_mse, 5)))
    testing_mape = mape(test_Y, fitted_test_Y)
    print("TESTING MAPE (%): " + str(testing_mape))

    linreg_Plots(test_Y, fitted_test_Y, resid_test_Y, "TEST_" + type_of_data_arg, home_directory)

if __name__ == "__main__": 
    ### Possble Choices for arguments to main() are:
        # NUMERIC
        # NUMERIC_AND_TEXT

    main("numeric") 
    main("numeric_and_text") 




