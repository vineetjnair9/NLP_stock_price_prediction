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
write_arima = 1
write_linreg = 1



def make_latex_table(dictionary, filename_addition, title, order_dict = {}, R_squared = 0): 
  caption = title
  label = title



  # print(dictionary)
  to_write_df = pd.DataFrame.from_dict(dictionary)
  # print(to_write_df)

  ### Make Dict and Convert to Latex ###
  if order_dict != {}:
    to_write_df['order'] = [order_dict[i] for i in to_write_df.index]
    to_write_df.sort_values(inplace = True, by = ['order'])
    to_write_df.drop('order', inplace = True, axis = 1)
  # print(to_write_df)
  

  # if R_squared == 1:
  #   print("in")
  #   new_columns = []
  #   for col in to_write_df.index:
  #     print(col)
  #     if "R2" in col:
  #       print("found")
  #       col = col.replace("R2", "$R^{2}$")
  #       new_columns.append(col)
  #     else:
  #       new_columns.append(col)
  #   to_write_df.index = new_columns

  # print(to_write_df)

  


  string = to_write_df.to_latex(bold_rows = False)


  ### Convert Rule Notation to hline Notation ###
  string = string.replace("toprule", "")
  string = string.replace("midrule", "")
  string = string.replace("bottomrule", "")

  ### Make Row Lines ###
  string = string.replace("\\\\", "\\\ \hline")


  ### Make Column Lines ###
  num_columns = len(dictionary.keys())+1
  to_find = "{" + "lr" + "}"
  to_replace = "{" + "|" + "c|"*num_columns + "}" + "\hline"
  string = string.replace(to_find, to_replace)


  ### Fix Title of First Column ###

  ### Fix Newlines ###
  string = string.split("\n" )
  string = list(filter(lambda a: a != "\\", string))
  string = "\n".join(string)

  ### Add Caption and Label ###
  table_string = "\\begin{table}[!h]\n" +  "\\caption{" + caption + "}\n" + " \\label{" + label + "}\n" + "\\captionsetup{justification=centering} \n" + " \\begin{center}" + string + " \\end{center}" + "\n" + "\\end{table}"
  # print(to_write_df)

  # if index_of_dict_to_pull == 2:
  with open(home_directory + "/LatexTables/Tables/" + filename_addition + ".tex", "w") as fileout:
    fileout.write(table_string)
    fileout.close()


def main(home_directory):
#------------------------------------## ARIMA ##------------------------------------#

  if write_arima == 1:
    directory = home_directory + "/ARIMA_PNGs/" 
    arima_numeric = pd.read_csv(directory + "Numeric_Results.csv", index_col = 0)
    arima_numeric.index.name = 'Ticker'
    arima_numeric = arima_numeric.transpose()
    arima_numeric_dict = arima_numeric.to_dict()
    title = "ARIMA Results - Numeric"
    make_latex_table(arima_numeric_dict, "Arima_Numeric_Results", title)


    directory = home_directory + "/ARIMA_PNGs/" 
    arima_numeric_and_text = pd.read_csv(directory + "Numeric_and_Text_Results.csv", index_col = 0)
    arima_numeric_and_text.index.name = 'Ticker'
    arima_numeric_and_text = arima_numeric_and_text.transpose()
    arima_numeric_and_text_dict = arima_numeric_and_text.to_dict()
    make_latex_table(arima_numeric_and_text_dict, "Arima_Numeric_and_Text_Results", title)


    title = "ARIMA MSE - Numeric and Text"
    ticker_dict = arima_numeric_dict
    ticker_dict[list(arima_numeric_dict.keys())[0]]["Testing, ARIMA and Regression, Numeric and Text"] = arima_numeric_and_text_dict[list(arima_numeric_and_text_dict.keys())[0]]["Testing, ARIMA and Regression, Numeric and Text"]
    ticker_dict[list(arima_numeric_dict.keys())[0]]["Training, ARIMA and Regression, Numeric and Text"] = arima_numeric_and_text_dict[list(arima_numeric_and_text_dict.keys())[0]]["Training, ARIMA and Regression, Numeric and Text"]
    order_dict = {'Testing, ARIMA and Regression, Numeric':5 ,
          'Testing, ARIMA and Regression, Numeric and Text': 6,
          'Testing, ARIMA, Numeric': 4,
          'Training, ARIMA and Regression, Numeric': 2,
          'Training, ARIMA and Regression, Numeric and Text': 3,
          'Training, ARIMA, Numeric': 1}
    make_latex_table(ticker_dict, "ARIMA_Results", title, order_dict)


#------------------------------------## LinReg ##------------------------------------#
  if write_linreg == 1:
    directory = home_directory + "/LinReg_Results/" 
    linreg_numeric = pd.read_csv(directory + "linreg_cv_percent_changes.csv", index_col = 0)
    linreg_numeric.index.name = 'Ticker'
    print(linreg_numeric)
    # linreg_numeric.columns = [i + ", Numeric" for i in linreg_numeric.columns]
    linreg_numeric = linreg_numeric.transpose()
    linreg_numeric_dict = linreg_numeric.to_dict()

    print(linreg_numeric_dict)
    ticker_dict = linreg_numeric_dict


    title = "Linear Regression Results - Percent Change Using Textual Features"
    order_dict = {'Percent Change in Training Adjusted R2':1
    , 'Percent Change in Testing R2':2
    , 'Percent Change in Training MSE':3
    , 'Percent Change in Testing MSE': 4}
    make_latex_table(ticker_dict, "LinearRegression_Results", title, order_dict, 1)

    home_directory + "/LatexTables/Tables/"
    

if __name__ == "__main__": 
    main(home_directory)



