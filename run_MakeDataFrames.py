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

start_date = datetime(2012, 9, 30)
start_date_string = str(start_date.month) + "/" + str(start_date.day) + "/" + str(start_date.year)
end_date = datetime(2020, 9, 30)
end_date_string = str(end_date.month) + "/" + str(end_date.day) + "/" + str(end_date.year)


#------------------------------------## GET RELEVANT NUMERIC DATA ##------------------------------------#
def main(type_of_data_arg, ticker_ARG):
    print("\n")
    print("Making DF for: " + type_of_data_arg + ", ticker: " + ticker_ARG + "...\n")

    home_directory = os.getcwd()

    if type_of_data_arg == "numeric":
        training_numeric_df = create_numeric_training_data(ticker_ARG, start_date, end_date)
        training_numeric_df['Date_Column'] = training_numeric_df.index.to_list()
        training_numeric_df = training_numeric_df[training_numeric_df['Date_Column'] <= datetime(2019, 12, 31)]
        training_numeric_df.drop(['Date_Column'], axis = 1, inplace = True)

        training_numeric_df.to_csv(home_directory + "/DataCSVs/" + type_of_data_arg + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv")


    elif type_of_data_arg == "numeric_and_text":

        ### Read in numeric data ###
        training_numeric_df = pd.read_csv(home_directory + "/DataCSVs/numeric_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_col = 0)

        ### Read in text data ###
        training_text_df = pd.read_csv(home_directory + "/DataCSVs/" + ticker_ARG + "_text_embs.csv", index_col = 0)
        training_text_df.index = [str(i.split("/")[2] + "-" + i.split("/")[0] + "-" + i.split("/")[1]) for i in training_text_df.index.to_list()]
        training_text_df = training_text_df.shift(1) # shift down 1 day


        ### Create joint df ###
        training_numeric_and_text_df = pd.merge(training_numeric_df, training_text_df, how='inner', left_index=True, right_index=True)
        # training_numeric_and_text_df['Date_Columns'] = training_numeric_and_text_df.index
        training_numeric_and_text_df.to_csv(home_directory + "/DataCSVs/" + type_of_data_arg + "_training_data_" + ticker_ARG + "_" + "_".join(start_date_string.split("/")) + "_" + "_".join(end_date_string.split("/"))  + ".csv", index_label = "Date_Column")


    elif type_of_data_arg == "article_embeddings":
        ### Read in embedding data ###
        text_df = pd.read_csv(home_directory + "/DataCSVs/" + ticker_ARG + "_text_embs_per_art.csv", index_col = 0)
        text_df['key'] = text_df['title'] + ", " + text_df['link'] #+ ", " + text_df['text']
        text_df = text_df.set_index('key')
        text_df.drop(['link', 'text', 'title', 'core_search_term', 'Unnamed: 0.1', 'id'], axis = 1, inplace = True)
        text_df.to_csv(home_directory + "/DataCSVs/" + ticker + "_article_TEXT_embeddings.csv")


        ### Read in embedding data ###
        title_df = pd.read_csv(home_directory + "/DataCSVs/" + ticker_ARG + "_title_embs_per_art.csv", index_col = 0)
        title_df['key'] = title_df['title'] + ", " + title_df['link'] #+ ", " + title_df['text']
        title_df = title_df.set_index('key')
        title_df.drop(['link', 'text', 'title', 'core_search_term', 'Unnamed: 0.1', 'id'], axis = 1, inplace = True)
        title_df.to_csv(home_directory + "/DataCSVs/" + ticker + "_article_TITLE_embeddings.csv")



if __name__ == "__main__":
    tickers = sys.argv[1].split(",")


    ### Possble Choices for arguments to main() are:
        # NUMERIC
        # NUMERIC_AND_TEXT
        # article_embeddings

    for ticker in tickers:
        # main("numeric", ticker)
        main("numeric_and_text", ticker)
        main("article_embeddings", ticker)



