# -*- coding: utf-8 -*-
"""

@author: mosta
"""

import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as prep
from sklearn.datasets import make_blobs
# from plotnine import *
# StandardScaler is a function to normalize the data
# You may also check MinMaxScaler and MaxAbsScaler
#from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import normalize
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from sklearn import datasets


######### Iris  ############

iris = datasets.load_iris()

# select first two columns
X = iris.data[:, :2]
max1= X[:,0].max()
X[:,0]=X[:,0]/max1

max2= X[:,1].max()
X[:,1]=X[:,1]/max2

# turn it into a dataframe
d = pd.DataFrame(X)

 
# plot the data
plt.scatter(d[0], d[1])

################### KMeans ##################
def display_cluster(X,km=[],num_clusters=0):
    color = 'brgcmyk' #List colors
    alpha = 0.5 #color obaque
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[km.labels_==i,0],X[km.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(km.cluster_centers_[i][0],km.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)

plt.rcParams['figure.figsize'] = [8,8]
sns.set_style("whitegrid")
sns.set_context("talk")
n_bins = 6
centers = [(-3, -3), (0, 0), (5,2.5),(-1, 4), (4, 6), (9,7)]

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 15)
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
 
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
    
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_
   
    
for key, val in mapping1.items():
    print(f'{key} : {val}')
plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


kmeans = KMeans(n_clusters=6).fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


range_n_clusters = [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14]
scores=[]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

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
    scores=np.append(scores,silhouette_avg)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0,ith_cluster_silhouette_values,facecolor=color,edgecolor=color,alpha=0.7,)

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
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0],centers[:, 1],marker="o",c="white",alpha=1,s=200,edgecolor="k",)

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"% n_clusters,fontsize=14,fontweight="bold",)
  
K_used=np.argmax(scores)+2
Best_score=np.max(scores)
plt.show()


###############   DBSCAN   ################

db = DBSCAN(eps=0.5,min_samples=5) 
db.fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
epsilon=np.arange(0.1,3,0.1)
min_samples=np.arange(5,25,1)

dbscan_score=[];
score=0;
k_clusters=[];
all_scores=[];
ep=[]
samples=[]
for s in min_samples:
    dbscan_score=[];
    for e in epsilon:
        db = DBSCAN(eps=e, min_samples=s).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # print(labels)

# Number of clusters in labels, ignoring noise if present.

        n_clusters_ = len(set(labels)) 
        # n_clusters_ = db.fit_predict(X)- (1 if -1 in labels else 0)
        # n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        if  n_clusters_!=1:
            print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
            score=metrics.silhouette_score(X, labels);
            dbscan_score.append(score)
            all_scores.append(score)
            k_clusters.append(n_clusters_)
            ep.append(e)
            samples.append(s)

max_score_index=np.argmax(all_scores)
best_e=ep[max_score_index]
best_s=samples[max_score_index]
plt.figure()
plt.plot(ep,all_scores)
plt.title('silouellte score with epsilon')
plt.figure()
plt.plot(samples,all_scores)
plt.title('silouellte score with samples')




##################  Hierarchal  ###################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
  
# Normalizing the data so that the data approximately 
# follows a Gaussian distribution
X_normalized = normalize(X_scaled)
  
# Converting the numpy array into a pandas DataFrame
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components = 2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

plt.figure(figsize =(8, 8))
plt.title('Visualising the data')
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))
distance_th=np.arange(0.1,2,1)
silhouette_scores = []
for i in range(np.size(distance_th)):
    H1=AgglomerativeClustering(n_clusters =None,affinity='euclidean',linkage='single',distance_threshold=distance_th[i])
    H1_p=H1.fit_predict(X_principal)
    H1_clusters=np.max(H1_p)+1
    if H1_clusters!=1:
        silhouette_scores.append(silhouette_score(X_principal, H1.fit_predict(X_principal)))
        plt.figure(figsize =(6, 6))
        plt.scatter(X_principal['P1'], X_principal['P2'], c = H1.fit_predict(X_principal), cmap ='rainbow')
        plt.show()

   
    H2=AgglomerativeClustering(n_clusters =None,affinity='manhattan',linkage='average',distance_threshold=distance_th[i])
    H2_p=H2.fit_predict(X_principal)
    H2_clusters=np.max(H2_p)+1
    if H2_clusters!=1:
        silhouette_scores.append(silhouette_score(X_principal, H2.fit_predict(X_principal)))
        plt.figure(figsize =(6, 6))
        plt.scatter(X_principal['P1'], X_principal['P2'], c = H2.fit_predict(X_principal), cmap ='rainbow')
        plt.show()
        print('dist',distance_th[i])
        print('sil_score',silhouette_scores)
   
    H3=AgglomerativeClustering(n_clusters = None,affinity='cosine',linkage='average',distance_threshold=distance_th[i])
    H3_p=H3.fit_predict(X_principal)
    H3_clusters=np.max(H3_p)+1
    if H3_clusters!=1:
        silhouette_scores.append(silhouette_score(X_principal, H3.fit_predict(X_principal)))
        plt.figure(figsize =(6, 6))
        plt.scatter(X_principal['P1'], X_principal['P2'], c = H3.fit_predict(X_principal), cmap ='rainbow')
        plt.show()
final_s=np.max(silhouette_scores)     # get corresponsding parameters


################ gaussian mixture  ####################
d = pd.DataFrame(X)

# plot the data
plt.scatter(d[0], d[1])
gmm = GaussianMixture(n_components = 3,covariance_type='full')
gmm.fit(d)
 
# Assign a label to each sample
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 
# plot three clusters in same plot
plt.figure()
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')

# plt.scatter(d[0], d[1])
gmm = GaussianMixture(n_components = 3,covariance_type='tied')
gmm.fit(d)
 
# Assign a label to each sample
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 
# plot three clusters in same plot
plt.figure()
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')


# plt.scatter(d[0], d[1])
gmm = GaussianMixture(n_components = 3,covariance_type='spherical')
gmm.fit(d)
 
# Assign a label to each sample
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 
# plot three clusters in same plot
plt.figure()
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')

# plt.scatter(d[0], d[1])
gmm = GaussianMixture(n_components = 3,covariance_type='diag')
gmm.fit(d)
 
# Assign a label to each sample
labels = gmm.predict(d)
d['labels']= labels
d0 = d[d['labels']== 0]
d1 = d[d['labels']== 1]
d2 = d[d['labels']== 2]
 
# plot three clusters in same plot
plt.figure()
plt.scatter(d0[0], d0[1], c ='r')
plt.scatter(d1[0], d1[1], c ='yellow')
plt.scatter(d2[0], d2[1], c ='g')



# Stratch dataset to get ellipsoid data
#X = np.dot(X,np.random.RandomState(0).randn(2,2))

x,y = np.meshgrid(np.sort(X[:,0]),np.sort(X[:,1]))
XY = np.array([x.flatten(),y.flatten()]).T

GMM = GaussianMixture(n_components=3).fit(X) # Instantiate and fit the model
print('Converged:',GMM.converged_) # Check if the model has converged
means = GMM.means_ 
covariances = GMM.covariances_


# Predict
Y = np.array([[0.5],[0.5]])
prediction = GMM.predict_proba(Y.T)
print(prediction)

# Plot   
fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.scatter(X[:,0],X[:,1])
ax0.scatter(Y[0,:],Y[1,:],c='orange',zorder=10,s=100)
for m,c in zip(means,covariances):
    multi_normal = multivariate_normal(mean=m,cov=c)
    ax0.contour(np.sort(X[:,0]),np.sort(X[:,1]),multi_normal.pdf(XY).reshape(len(X),len(X)),colors='black',alpha=0.3)
    ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)
    
plt.title('Contouring')    
plt.show()
Best_score=np.max(scores)
best_score=all_scores[max_score_index]
final_s=np.max(silhouette_scores)
print('best score of kmeans',Best_score)
print('best score of DBSCAN',best_score)
print('best score of Hierarchal',final_s)
#comments:
    #the best score resulted in using heirarichal clustering technique 
    #the score resulted using this technique is 0.651 using affinity='manhattan',linkage='average', dist_threshold=0.1
    #the score of DBscan score is resulted when epsilon was 0.1 and number of samples is 14
