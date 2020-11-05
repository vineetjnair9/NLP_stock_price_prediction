from scraper_module import *
from get_financial_data import * 
from import_my_packages import * 
from featurizer_module import *
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


ticker = 'AMZN'
start_date = datetime(2019, 10, 30)
start_date_string = "10/30/2019" #2019/10/30"
end_date = datetime(2019, 11,5)
end_date_string = "11/05/2019" #2019/11/5"


get_numeric_data = 0
get_articles_data = 0

if get_numeric_data == 1:
	training_numeric_df = create_numeric_training_data(ticker, start_date, end_date)

if get_articles_data == 1:
	s = scraper(search_terms = ['MSFT'], date_from = start_date_string, date_to = end_date_string) #init
	s.make_df() #creates df, access with s.df
	s.df.to_csv(home_directory + "/Articles_K_Means/articles_for_kmeans_micro.csv")


# I gotta use scrape period. 
# s.scrape_period


# test = pd.DataFrame([["11/05/2020","ge","link"  "CHALFONT ST GILES, England--(BUSINESS WIRE)--T...  General Electric's stock surged" ]], columns = )


df = pd.read_csv(home_directory + "/Articles_K_Means/articles_for_kmeans.csv", index_col = 0)
df2 = pd.read_csv(home_directory + "/Articles_K_Means/articles_for_kmeans_micro.csv", index_col = 0)

df = df.append(df2)
test = pd.DataFrame([["11/05/2020","ge","link",  "some words" ], ["11/05/2020","ge","link2",  "some words 2222222" ]], columns = df.columns)
df = df.append(test)
# print(df)




df.drop_duplicates(inplace = True)
df = df.dropna()
df.reset_index(inplace = True)
df = df.drop(['index'], axis = 1)
dates = set(df['date'].copy())

#### NEED TO TOKENIZE AND CLEAN HERE
df["text"] = df["text"].apply(preprocess)
print(df)


out = ' '.join(df["text"])
out = preprocess(preprocess(out))
print(out)
# out = out.split(" ")


# unique_words = list(set(out))
# count_list = [out.count(i) for i in unique_words]
# count_dict = {unique_words[i]: count_list[i] for i in range(len(unique_words))}
# count_dict = list(count_dict.items())
# count_dict.sort(key = lambda x: x[1], reverse = True) 


# from collections import Counter
# import numpy as np
# import matplotlib.pyplot as plt


# counts = dict(Counter(out).most_common(10))

# labels, values = zip(*counts.items())

# # sort your values in descending order
# indSort = np.argsort(values)[::-1]

# # rearrange your data
# labels = np.array(labels)[indSort]
# values = np.array(values)[indSort]

# indexes = np.arange(len(labels))

# bar_width = 0.35

# plt.bar(indexes, values)

# # add labels
# plt.xticks(indexes + bar_width, labels)
# plt.show()



from wordcloud import WordCloud
# Generate word cloud
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='salmon', colormap='Pastel1', collocations=False).generate(out)
# Plot
plot_cloud(wordcloud)



fig = plt.figure(figsize = (40,22))
plt.plot([c[1] for c in count_dict ], linewidth = 3, color = 'orangered' )
plt.title("Word Frequency Plot", fontdict = font_dict)
plt.xlabel("Word", fontdict = font_dict)
plt.ylabel("Number of Instances", fontdict = font_dict)
ax = plt.gca()
labels = [item.get_text() for item in ax.get_xticklabels()]
print(labels)
for i in range(len(labels)):
    labels[i] = count_dict[i][0]
ax.set_xticklabels(labels)
plt.xticks(rotation = 90)
plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
fig.savefig(home_directory + "/Articles_K_Means/Figures/WordFreqPlot.jpg", bbox_inches="tight")
plt.close()

print([c[0] for c in count_dict])
# fig = plt.figure(figsize = (10,4))
# plt.gcf().subplots_adjust(bottom=0.15) # to avoid x-ticks cut-off

# fd = nltk.FreqDist(nltk.tokenize.word_tokenize(out))
# fd.plot(100,cumulative=False)
# plt.figure(figsize=(20, 8))
# plt.title("Word Frequency", fontdict = font_dict)
# plt.xticks(fontsize = tick_size, fontname = font_dict['family'])
# plt.yticks(fontsize = tick_size, fontname = font_dict['family'])
# plt.tight_layout()
# fig.savefig(home_directory + "/Articles_K_Means/Figures/WordFreqPlot.jpg", bbox_inches="tight")
# plt.close()


f = Featurizer()

print(df)

### Make tfdid Training Matrix ###
f = Featurizer()
for a in df['text']:
    f.preprocess(a)

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















# from sklearn.decomposition import PCA

# def init_medoids(X, k):
#     from numpy.random import choice
#     from numpy.random import seed
 
#     seed(1)
#     samples = choice(len(X), size=k, replace=False)
#     return X[samples, :]



# def compute_d_p(X, medoids, p):
#     m = len(X)
#     medoids_shape = medoids.shape
#     # If a 1-D array is provided, 
#     # it will be reshaped to a single row 2-D array
#     if len(medoids_shape) == 1: 
#         medoids = medoids.reshape((1,len(medoids)))
#     k = len(medoids)
#     S = np.empty((m, k))
#     for i in range(m):
#         d_i = np.linalg.norm(X[i, :] - medoids, ord=p, axis=1)
#         S[i, :] = d_i**p
#     return S


# def has_converged(old_medoids, medoids):
#     return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])
  
# #Full algorithm
# def kmedoids(X, k, p, starting_medoids=None, max_steps=np.inf):
#     if starting_medoids is None:
#         medoids = init_medoids(X, k)
#     else:
#         medoids = starting_medoids
        
#     converged = False
#     labels = np.zeros(len(X))
#     i = 1
#     while (not converged) and (i <= max_steps):
#         old_medoids = medoids.copy()
        
#         S = compute_d_p(X, medoids, p)
        
#         labels = assign_labels(S)
        
#         medoids = update_medoids(X, medoids, p)
        
#         converged = has_converged(old_medoids, medoids)
#         i += 1
#     return (medoids,labels)







# KMEDIAN
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn import datasets
# from sklearn.decomposition import PCA

# Dataset
# iris = datasets.load_iris()
# data = pd.DataFrame(iris.data,columns = iris.feature_names)

# target = iris.target_names
# labels = iris.target

# #Scaling
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# #PCA Transformation
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(data)
# PCAdf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2','principal component 3'])

# datapoints = PCAdf.values
# m, f = datapoints.shape
# k = 3

# #Visualization
# fig = plt.figure(1, figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)
# X_reduced = points
# ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels,
#            cmap=plt.cm.Set1, edgecolor='k', s=40)
# ax.set_title("First three PCA directions")
# ax.set_xlabel("principal component 1")
# ax.w_xaxis.set_ticklabels([])
# ax.set_ylabel("principal component 1")
# ax.w_yaxis.set_ticklabels([])
# ax.set_zlabel("principal component 1")
# ax.w_zaxis.set_ticklabels([])
# plt.show()

# def init_medoids(X, k):
#     from numpy.random import choice
#     from numpy.random import seed
 
#     seed(1)
#     samples = choice(len(X), size=k, replace=False)
#     return X[samples, :]

# # medoids_initial = init_medoids(datapoints, 3)

# def compute_d_p(X, medoids, p):
#     m = len(X)
#     medoids_shape = medoids.shape
#     # If a 1-D array is provided, 
#     # it will be reshaped to a single row 2-D array
#     if len(medoids_shape) == 1: 
#         medoids = medoids.reshape((1,len(medoids)))
#     k = len(medoids)
    
#     S = np.empty((m, k))
    
#     for i in range(m):
#         d_i = np.linalg.norm(X[i, :] - medoids, ord=p, axis=1)
#         S[i, :] = d_i**p

#     return S
  
# # S = compute_d_p(datapoints, medoids_initial, 2)


# def assign_labels(S):
#     return np.argmin(S, axis=1)
  
# # labels = assign_labels(S)

# def update_medoids(X, medoids, p):
    
#     S = compute_d_p(datapoints, medoids, p)
#     labels = assign_labels(S)
        
#     out_medoids = medoids
                
#     for i in set(labels):
        
#         avg_dissimilarity = np.sum(compute_d_p(datapoints, medoids[i], p))

#         cluster_points = datapoints[labels == i]
        
#         for datap in cluster_points:
#             new_medoid = datap
#             new_dissimilarity= np.sum(compute_d_p(datapoints, datap, p))
            
#             if new_dissimilarity < avg_dissimilarity :
#                 avg_dissimilarity = new_dissimilarity
                
#                 out_medoids[i] = datap
                
#     return out_medoids

# def has_converged(old_medoids, medoids):
#     return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])
  
# #Full algorithm
# def kmedoids(X, k, p, starting_medoids=None, max_steps=np.inf):
#     if starting_medoids is None:
#         medoids = init_medoids(X, k)
#     else:
#         medoids = starting_medoids
        
#     converged = False
#     labels = np.zeros(len(X))
#     i = 1
#     while (not converged) and (i <= max_steps):
#         old_medoids = medoids.copy()
        
#         S = compute_d_p(X, medoids, p)
        
#         labels = assign_labels(S)
        
#         medoids = update_medoids(X, medoids, p)
        
#         converged = has_converged(old_medoids, medoids)
#         i += 1
#     return (medoids,labels)

# #Count
# def mark_matches(a, b, exact=False):
#     """
#     Given two Numpy arrays of {0, 1} labels, returns a new boolean
#     array indicating at which locations the input arrays have the
#     same label (i.e., the corresponding entry is True).
    
#     This function can consider "inexact" matches. That is, if `exact`
#     is False, then the function will assume the {0, 1} labels may be
#     regarded as the same up to a swapping of the labels. This feature
#     allows
    
#       a == [0, 0, 1, 1, 0, 1, 1]
#       b == [1, 1, 0, 0, 1, 0, 0]
      
#     to be regarded as equal. (That is, use `exact=False` when you
#     only care about "relative" labeling.)
#     """
#     assert a.shape == b.shape
#     a_int = a.astype(dtype=int)
#     b_int = b.astype(dtype=int)
#     all_axes = tuple(range(len(a.shape)))
#     assert ((a_int == 0) | (a_int == 1) | (a_int == 2)).all()
#     assert ((b_int == 0) | (b_int == 1) | (b_int == 2)).all()
    
#     exact_matches = (a_int == b_int)
#     if exact:
#         return exact_matches

#     assert exact == False
#     num_exact_matches = np.sum(exact_matches)
#     if (2*num_exact_matches) >= np.prod (a.shape):
#         return exact_matches
#     return exact_matches == False # Invert

# def count_matches(a, b, exact=False):
#     """
#     Given two sets of {0, 1} labels, returns the number of mismatches.
    
#     This function can consider "inexact" matches. That is, if `exact`
#     is False, then the function will assume the {0, 1} labels may be
#     regarded as similar up to a swapping of the labels. This feature
#     allows
    
#       a == [0, 0, 1, 1, 0, 1, 1]
#       b == [1, 1, 0, 0, 1, 0, 0]
      
#     to be regarded as equal. (That is, use `exact=False` when you
#     only care about "relative" labeling.)
#     """
#     matches = mark_matches(a, b, exact=exact)
#     return np.sum(matches)



# def get_df_index_of_spcific_row(numpydf, mediods):
#     numpylist = numpydf.tolist()
#     print(numpylist)
#     indices = []
#     for i in range(len(mediods)):
#         print(mediods[i].tolist())
#         print("\n" + str(numpylist.index(mediods[i].tolist())))
#         indices.append(numpylist.index(mediods[i].tolist()))
#     return(indices)

# num_clusters = 6

# doc_matrix = np.round_(doc_matrix, decimals = 6)
# doc_matrix = doc_matrix * 1000000
# doc_matrix = doc_matrix.astype(int)
# print(doc_matrix)
# initial_medoids = doc_matrix[0:num_clusters]
# initial_medoids = initial_medoids.to_numpy()
# doc_matrix_df = doc_matrix.copy()
# doc_matrix = doc_matrix.to_numpy()



# data= doc_matrix_df.copy()

# datapoints= doc_matrix
# results = kmedoids(doc_matrix, 3, 2, starting_medoids = initial_medoids)
# mediods = results[0]
# assignment_vector = results[1]
# print("mediods")
# print(mediods)

# indices = get_df_index_of_spcific_row(doc_matrix, mediods)

# print(indices)
