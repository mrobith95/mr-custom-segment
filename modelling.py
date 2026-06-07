import pandas as pd
import os
from sklearn.cluster import KMeans

## read data for modelling and for rfm input
data     = pd.read_csv('data/feat_eng/data.csv', index_col='customer_id') ## read data
data_rfm = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id')

print(data.head(10))
print(data.info())

## define model
kproto = KMeans(n_clusters=4, ## number of clusters
                random_state=300 ## ensure replication
                )

## fit and cluster data
kproto.fit(data)
clus = kproto.predict(data)

## append clustering result, and save
data_rfm['cluster'] = clus

if not os.path.exists('data/modelling'):
    os.makedirs('data/modelling')

data_rfm.to_csv('data/modelling/data_pure_rfm.csv')