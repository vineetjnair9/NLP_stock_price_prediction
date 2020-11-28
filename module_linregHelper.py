from module_scraper import *
from module_numericFinancialData import * 
from import_my_packages import * 
from module_featurizer import *

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

def sse(target, fit, round_digits = 5):
    target = target.tolist()
    fit = list(fit)
    x = sum([(target[i] - fit[i])**2 for i in range(len(target))])
    x = round(x, round_digits)
    return(x) 


def mse(target, fit, round_digits = 5):
    target = target.tolist()
    fit = list(fit)
    x = statistics.mean([(target[i] - fit[i])**2 for i in range(len(target))])
    x = round(x, round_digits)
    return(x) 


def mape(target, fit, round_digits = 5):
    target = target.tolist()

    fit = list(fit)
    x = statistics.mean([100 * (abs(target[i] - fit[i])/target[i]) for i in range(len(target))])
    x = round(x, round_digits)
    return(x) # returns MAPE as a PERCENT

def percent_change(new, old, r_squared_indicator = 0):
    if r_squared_indicator == 1:
        return(round(100 * ((new - old)/abs(old)), 4))
    else:
        return(round(100 * ((new - old)/old), 4))


def vif(df):
    vif_data = pd.DataFrame() 
    vif_data["feature"] = df.columns 
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))] 
    return(vif_data)



def plot_regularization_hyperparameters(l1_wts_to_test, hyperparameters_results, home_directory, ticker):
    fig = plt.figure(figsize = (20,12))
    plt.plot(l1_wts_to_test, hyperparameters_results, linewidth = 5, markersize = 20, color = 'orangered')
    plt.title("Hyperparameter Tuning - Regularization", fontdict = font_dict)
    plt.xlabel("l1wt Value", fontdict = font_dict)
    plt.ylabel("Validation MSE", fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_L1_WTS_Parameter_Validation_Results.jpg", bbox_inches="tight")



def correlation_plot(X, text_extention, home_directory, ticker):
    corr = X.corr(method='pearson') 
    fig, ax = plt.subplots(figsize=(30,30))
    sns.heatmap(corr, annot=True, xticklabels=corr.columns, 
            yticklabels=corr.columns, ax=ax, linewidths=.5, 
            vmin = -1, vmax=1, center=0, square = False)
    plt.title('Correlation HeatMap for ALL DATA')
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Correlation_Test_" + text_extention + ".jpg", bbox_inches="tight")



def linreg_Plots(true_y, fitted_y, residuals, text_extension, home_directory, ticker):
    fig = plt.figure(figsize = (20,12))
    plt.scatter(fitted_y, residuals, s = 100)
    plt.title("Residuals vs. Fit - " + text_extension, fontdict = font_dict)
    plt.xlabel("Fitted Values", fontdict = font_dict)
    plt.ylabel("Residuals", fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.axhline(y=0,color='gray',linestyle='--', linewidth = 3)
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Linear_Regression_Numeric_" + text_extension + "_Residuals_vs_Fitted.jpg", bbox_inches="tight")



    fig = plt.figure(figsize = (20,12))
    plt.scatter(true_y, residuals, s = 100)
    plt.title("Residuals vs. Target - " + text_extension, fontdict = font_dict)
    plt.xlabel("Target Values", fontdict = font_dict)
    plt.ylabel("Residuals", fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.axhline(y=0,color='gray',linestyle='--', linewidth = 3)
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Linear_Regression_Numeric_" + text_extension + "_Residuals_vs_Target.jpg", bbox_inches="tight")


    fig = plt.figure(figsize = (20,20))
    plt.scatter(true_y, fitted_y, s = 100)
    plt.title("Fitted vs. Target - " + text_extension, fontdict = font_dict)
    plt.xlabel("Target Values", fontdict = font_dict)
    plt.ylabel("Fitted Values", fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    ax = plt.gca()
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Linear_Regression_Numeric_" + text_extension + "_Fitted_vs_Target.jpg", bbox_inches="tight")

    fig = plt.figure(figsize = (20,12))
    plt.hist(true_y, bins= 20)
    plt.title("Histogram of True Target Values - " + text_extension, fontdict = font_dict)
    plt.xlabel("Target Values", fontdict = font_dict)
    plt.ylabel("Frequency", fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.axhline(y=0,color='gray',linestyle='--', linewidth = 3)
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Linear_Regression_Numeric_" + text_extension + "_Histogram_of_True_Target_Values.jpg", bbox_inches="tight")


    fig = plt.figure(figsize = (20,12))
    plt.hist(residuals, bins= 20)
    plt.title("Histogram of Residuals - " + text_extension, fontdict = font_dict)
    plt.xlabel("Regression Residuals", fontdict = font_dict)
    plt.ylabel("Frequency", fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.axhline(y=0,color='gray',linestyle='--', linewidth = 3)
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Linear_Regression_Numeric_" + text_extension + "_Histogram_of_Regression_Residuals.jpg", bbox_inches="tight")


def plot_metrics_for_many_iterations(training_adj_r_squared, training_mse_list, testing_mse_list, testing_adj_r_squared, text_extension, home_directory, ticker):
    # fig = plt.figure(figsize = (22,15))
    labelsize = 20

    fig, ax1 = plt.subplots()
    plt.title("Cross Validation for " + ticker, fontdict = font_dict)
    color = 'tab:red'
    ax1.set_xlabel('Iteration', fontdict = font_dict)
    ax1.set_ylabel('$R^2$', color=color, fontdict = font_dict)
    ax1.plot(training_adj_r_squared, label = "Training R^2", markersize = 100, linewidth = 4, linestyle = "dashed", color=color)
    plt.plot(testing_adj_r_squared, label = "Testing R^2", markersize = 100, linewidth = 4, linestyle = "solid", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='both', which='major', labelsize=labelsize)
    ax1.tick_params(axis='both', which='minor', labelsize=labelsize)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('MSE', color=color, fontdict = font_dict)  # we already handled the x-label with ax1
    ax2.plot(training_mse_list, label = "Training Negative MSE", markersize = 100, linewidth = 4, linestyle = "dashed", color=color)
    ax2.plot(testing_mse_list, label = "Testing Negative MSE", markersize = 100, linewidth = 4, linestyle = "solid", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax2.tick_params(axis='both', which='major', labelsize=labelsize)
    ax2.tick_params(axis='both', which='minor', labelsize=labelsize)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped


    fig = plt.gcf()
    fig.set_size_inches(40, 25)
    fig.legend(prop=font_dict)
    fig.savefig(home_directory + "/LinReg_Results/Figures/" + ticker + "_Linear_Regression_" + text_extension + "_Cross_Validation.jpg", bbox_inches="tight")


