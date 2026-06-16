import pandas as pd
import os
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans

## read data for modelling and for rfm input
data     = pd.read_csv('data/feat_eng/data.csv', index_col='customer_id') ## read data
data_rfm = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id')

print(data.head(10))
print(data.info())

## fit and predict data
kproto = KPrototypes(n_clusters=6, ## number of clusters
                     random_state=300, ## ensure replication
                     n_init=1, ## n_init=10 takes so long huh
                     verbose=2 ## verbose printing
                     )
## good news is gamma (weighting factor) are determined from data
## initializer follow Cao's paper listed in pypi page

kproto.fit(data, categorical=[i for i in range(7,21)])
clus = kproto.predict(data, categorical=[i for i in range(7,21)])

# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# # Print training statistics
# print(kproto.cost_)
# print(kproto.n_iter_)

## append clustering result, and save
data_rfm['cluster'] = clus

if not os.path.exists('data/modelling'):
    os.makedirs('data/modelling')

data_rfm.to_csv('data/modelling/data.csv')