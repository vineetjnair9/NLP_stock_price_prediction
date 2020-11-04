from scraper_module import *
from get_financial_data import * 
from import_my_packages import * 
from featurizer_module import *
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
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
	s = scraper(search_terms = ['GE'], date_from = start_date_string, date_to = end_date_string) #init
	s.make_df() #creates df, access with s.df
	print(s.df)
	s.df.to_csv(home_directory + "/Articles_K_Means/articles_for_kmeans.csv")


# I gotta use scrape period. 
# s.scrape_period


# test = pd.DataFrame([["11/05/2020","ge","link"  "CHALFONT ST GILES, England--(BUSINESS WIRE)--T...  General Electric's stock surged" ]], columns = )


df = pd.read_csv(home_directory + "/Articles_K_Means/articles_for_kmeans.csv", index_col = 0)
test = pd.DataFrame([["11/05/2020","ge","link",  "some text" ], ["11/05/2020","ge","link2",  "some text 2222222" ]], columns = df.columns)
df = df.append(test)
# print(df)


df.drop_duplicates(inplace = True)
df = df.dropna()
df.reset_index(inplace = True)
df = df.drop(['index'], axis = 1)
dates = set(df['date'].copy())

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

print(elbow)
print(ks)


from sklearn.metrics import silhouette_samples, silhouette_score


num_clusters =2
### Run KMeans ###
kmeans = KMeans(n_clusters=num_clusters).fit(doc_matrix)
cluster_assignment_vector = kmeans.fit_predict(doc_matrix)
center_list = kmeans.cluster_centers_.tolist()
silhouette_avg = silhouette_score(doc_matrix, cluster_assignment_vector)
print("For n_clusters =", num_clusters,
          "The average silhouette_score is :", silhouette_avg)

X = doc_matrix
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
    # ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors, edgecolor='k')

    # Labeling the clusters
    # centers = clusterer.cluster_centers_
    # # Draw white circles at cluster centers
    # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
    #             c="white", alpha=1, s=200, edgecolor='k')

    # for i, c in enumerate(centers):
    #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
    #                 s=50, edgecolor='k')

    # ax2.set_title("The visualization of the clustered data.")
    # ax2.set_xlabel("Feature space for the 1st feature")
    # ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    fig.savefig(home_directory + "/Articles_K_Means/Figures/silh_" + str(i) + ".jpg")
    plt.close()

# plt.show()














# #### Make Corpus ####
# for i in df.index:
# 	print(i)
# 	a = df.at[i, 'text']
# 	print(a)
# 	df.set_value(i, 'text', preprocess(a[0])) 
# 	f.preprocess(a[0])
# 	print("\n")


# print('pre-processed articles/text:', f.corpus)
# print(df)

# output_df = pd.DataFrame(columns = ['date', 'core_search_term', 'all_articles_text'])
# for d in dates:
# 	print(d)
# 	tiny_df = df[df['date'] == d]
# 	tiny_df = tiny_df.drop(['link'],axis = 1)
# 	tiny_df['all_articles_text'] = tiny_df['text'].str.cat(sep=', ')
# 	tiny_df = tiny_df.drop(['text'],axis = 1)
# 	print(tiny_df)
# 	print("\n")
# 	output_df = output_df.append(tiny_df.iloc[0])

# print(output_df)






# # training_corpus = ['this is information about document one', 
# #                    '$3NoW, so!me^ information about the >,/?seco0nd! 488492document',
# #                   'this is the last article news!!']

# # print('')
# doc_matrix = f.tfidf_fit(use_idf = True)
# doc_matrix
# #new testing examples to be used as features
# test_corpus = ['one test instance', 
#                'this is the second one, hopefully contains some information',
#               'Final $#document742 with [information]!}']

# f.tfidf_transform(test_corpus)



# print(preprocess('This&/ $!!is a #42 {[important]} +70TEST|'))