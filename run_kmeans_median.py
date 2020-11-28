from module_scraper import *
from module_numericFinancialData import *
from import_my_packages import *
from module_featurizer import *
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
# from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import spatial
##################### PLOT SETTINGS #####################
font_dict = {'size' : 40, 'family': 'serif'}
font_dict_legend = {'size' : 20, 'family': 'serif'}
tick_size = 30
###################################################################

home_directory = os.getcwd()
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

#------------------------------------## SPECIFY SETTINGS ##------------------------------------#

ticker = 'AMZN'
start_date = datetime(2019, 10, 30)
start_date_string = str(start_date.month) + "/" + str(start_date.day) + "/" + str(start_date.year)
end_date = datetime(2020, 10,30)
end_date_string = str(end_date.month) + "/" + str(end_date.day) + "/" + str(end_date.year)

#------------------------------------## BEGIN ##------------------------------------#


get_articles_data = 0

if get_articles_data == 1:
	s = scraper(search_terms = ['MSFT'], date_from = start_date_string, date_to = end_date_string) #init
	s.make_df() #creates df, access with s.df
	s.df.to_csv(home_directory + "/Articles_K_Means/articles_for_kmeans_micro.csv")

df = pd.read_csv(home_directory + "/Articles_K_Means/articles_for_kmeans.csv", index_col = 0)
df2 = pd.read_csv(home_directory + "/Articles_K_Means/articles_for_kmeans_micro.csv", index_col = 0)

df = df.append(df2)
test = pd.DataFrame([["11/05/2020","ge","link",  "some words" ], ["11/05/2020","ge","link2",  "some words 2222222" ]], columns = df.columns)
df = df.append(test)


df.drop_duplicates(inplace = True)
df = df.dropna()
df.reset_index(inplace = True)
df = df.drop(['index'], axis = 1)
dates = set(df['date'].copy())

print(df)


f = Featurizer()

print(df)

### Make tfdid Training Matrix ###
f = Featurizer()
for a in df['text']:
    f.preprocess(a)
print(df)


# f.preprocess(df['text'])

# print('pre-processed articles/text:', f.corpus)
print('')
doc_matrix = f.tfidf_fit(use_idf = True)
print(doc_matrix)


### Make tfdid Testing Matrix ###
#new testing examples to be used as features

# f.tfidf_transform(test_corpus) # test_corpus = column name of testing df with article text



### Find Score for Different K's ###
def calculate_WSS(points, kmax):
	sse = []
	ks = []
	for k in range(1, kmax+1):
		ks.append(k)
		kmeans = KMeans(n_clusters = k).fit(points)
		centroGood_Market_IDs = kmeans.cluster_centers_
		pred_clusters = kmeans.predict(points)
		curr_sse = 0
		for i in range(len(points)):
			curr_center = centroGood_Market_IDs[pred_clusters[i]]
			point = points.iloc[i]
			# point = np.asarray([float(j) for j in points[i]])
			dist = scipy.spatial.distance.cdist([point], [curr_center], 'euclidean')[0][0]
			curr_sse = curr_sse + dist
		sse.append(curr_sse)
	return(sse, ks)



kmeans = 1
silhouette_scores = 0

if kmeans == 1:
    clusters_to_test = len(doc_matrix)-1
    elbow, ks = calculate_WSS(doc_matrix, clusters_to_test) # ks = number of clusters
    fig = plt.figure(figsize = (22,15))
    plt.plot(ks, elbow, linewidth = 3, color = 'orangered')#, c=colors[1])
    plt.title("Elbow Chart to Determine Best K for KMeans Clustering", fontdict = font_dict)
    plt.xlabel("K Value",fontdict = font_dict)
    plt.ylabel("Total Euclidean Distance",fontdict = font_dict)
    plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
    plt.tight_layout()
    fig.savefig(home_directory + "/Articles_K_Means/Figures/Optimal_K_Value.jpg", bbox_inches="tight")
    plt.close()




    num_clusters =2
    ### Run KMeans ###
    kmeans = KMeans(n_clusters=num_clusters).fit(doc_matrix)
    cluster_assignment_vector = kmeans.fit_predict(doc_matrix)
    center_list = kmeans.cluster_centers_.tolist()
    # print("Centers:")
    # print(center_list)

    silhouette_avg = silhouette_score(doc_matrix, cluster_assignment_vector)
    print("For n_clusters =", num_clusters,
              "The average silhouette_score is :", silhouette_avg)



    cosine_sim = np.ones((len(doc_matrix), len(center_list)))
    for article_num in range(len(doc_matrix)):
        for center_num in range(len(center_list)):
            # print(article_num)
            # print(center_num)
            result = 1 - spatial.distance.cosine(center_list[center_num], doc_matrix.loc[article_num])
            cosine_sim[article_num][center_num] = result

    print(cosine_sim)
    max_indeices = cosine_sim.argmax(axis=0)
    print(max_indeices)
    Articles_closest_to_centers = {}
    for center_num in range(len(center_list)):
        Articles_closest_to_centers[str(center_num)] = df.at[max_indeices[center_num], 'text'] # replace 'text' with header

    pp.pprint(Articles_closest_to_centers)

    X = doc_matrix

    if silhouette_scores == 1:
        for n_clusters in range(2, clusters_to_test):
            # Create a subplot with 1 row and 2 columns
            fig = plt.figure(figsize = (15,10)) #plt.subplots(1, 1)
            ax1 = plt.gca()
            # fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()
                print(ith_cluster_silhouette_values)

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)


            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
            fig.savefig(home_directory + "/Articles_K_Means/Figures/silh_" + str(i) + ".jpg")
            plt.close()













