import pandas as pd
from kmodes.kprototypes import KPrototypes
import os
from sklearn.cluster import KMeans

## read data for modelling and for rfm input
data     = pd.read_csv('data/feat_eng/data.csv', index_col='customer_id') ## read data
data_rfm = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id')

print(data.head(10))
print(data.info())

## fit and predict data
# kproto = KPrototypes(n_clusters=6, ## number of clusters
#                      random_state=300, ## ensure replication
#                      n_init=1, ## n_init=10 takes so long huh
#                      verbose=2 ## verbose printing
#                      )
kproto = KMeans(n_clusters=4, ## number of clusters
                random_state=300 ## ensure replication
                )
## good news is gamma (weighting factor) are determined from data
## initializer follow Cao's paper listed in pypi page

kproto.fit(data)
clus = kproto.predict(data)

# # Print cluster centroids of the trained model.
# print(kproto.cluster_centroids_)
# # Print training statistics
# print(kproto.cost_)
# print(kproto.n_iter_)

## append clustering result, and save
data_rfm['cluster'] = clus

if not os.path.exists('data/modelling'):
    os.makedirs('data/modelling')

data_rfm.to_csv('data/modelling/data_pure_rfm.csv')

# note: cost per cluster
#  2 -> 61227.509
#  3 -> 53755.750
#  4 -> 48735.736
#  5 -> 45066.034
#  6 -> 41651.084

# ## make dataframe for the centroids
# hasil = dict()
# hasil['Recency'] = kproto.cluster_centroids_[:,1]
# hasil['Frequency'] = kproto.cluster_centroids_[:,2]
# hasil['Monetary'] = kproto.cluster_centroids_[:,0]
# hasil['Size'] = kproto.cluster_centroids_[:,3]
# hasil['Custom Rate'] = kproto.cluster_centroids_[:,4]
# hasil['Weekend'] = kproto.cluster_centroids_[:,5]
# hasil['Time'] = kproto.cluster_centroids_[:,6]
# hasil_pd = pd.DataFrame(hasil)

# ## compute inverse transform
# mr_centroid = trf['mr_tf'].inverse_transform(hasil_pd[['Monetary', 'Recency']])
# fsc_centroid = trf['fsc_tf'].inverse_transform(hasil_pd[['Frequency', 'Size', 'Custom Rate']])

# print(mr_centroid)
# print(fsc_centroid)

