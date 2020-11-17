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


# tickers = ['AMZN', 'GEr', 'PFE', 'T', 'GS']
tickers = ['PFE', 'T', 'GS']
#------------------------------------## SPECIFY SETTINGS ##------------------------------------#


# ticker = 'AMZN'
start_date = datetime(2019, 10, 30)
start_date_string = str(start_date.month) + "/" + str(start_date.day) + "/" + str(start_date.year)
end_date = datetime(2020, 10,30)
end_date_string = str(end_date.month) + "/" + str(end_date.day) + "/" + str(end_date.year)


#------------------------------------## GET RELEVANT NUMERIC DATA ##------------------------------------#
def main(type_of_data_arg, ticker_ARG):
    print("\n")
    print("Making DF for: " + type_of_data_arg + ", ticker: " + ticker_ARG + "...\n")

    home_directory = os.getcwd()

    try:
        X = pd.read_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)
    except:
        print("DF requested does not yet exist. Writing new data now...")
        if type_of_data_arg == "numeric":
            training_numeric_df = create_numeric_training_data(ticker_ARG, start_date, end_date)
            training_numeric_df.to_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv")

        elif type_of_data_arg == "numeric_and_text":

            ### Write numeric data ###
            try: # if file already exists
                training_numeric_df = pd.read_csv(home_directory + "/numeric_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)
            except:
                training_numeric_df = create_numeric_training_data(ticker_ARG, start_date, end_date)
                training_numeric_df.to_csv(home_directory + "/" + "numeric" + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv")
                training_numeric_df = pd.read_csv(home_directory + "/numeric_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)

            ### Write text data ###
            try: # if file already exists
                training_text_df = pd.read_csv(home_directory + "/text_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)
            except:
###### NEED TO FIX TEXTUAL FEATURES HERE ####
                training_text_data = pd.DataFrame([['88888888', '99999999' ] for i in range(len(training_numeric_df.index))], columns = ['textual_feature_1', 'textual_feature_2']) # NEED TO SPECIFY SOME FUNCTION HERE
                training_text_data.index = training_numeric_df.index
###### NEED TO FIX TEXTUAL FEATURES HERE ####
                training_text_data.to_csv(home_directory + "/" + "text" + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv")
                training_text_df = pd.read_csv(home_directory + "/text_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)

            ### Create joint df ###
            training_numeric_and_text_df = pd.merge(training_numeric_df, training_text_df, how='outer', left_index=True, right_index=True)
            training_numeric_and_text_df.to_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv")

        X = pd.read_csv(home_directory + "/" + type_of_data_arg + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)

    print("Entire X DF:")
    print(X)


if __name__ == "__main__":
    ### Possble Choices for arguments to main() are:
        # NUMERIC
        # NUMERIC_AND_TEXT

    for ticker in tickers:
        main("numeric", ticker)
        # main("numeric_and_text", ticker)




