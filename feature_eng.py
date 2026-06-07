import pandas as pd
import os
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
import skops.io as sio

## read data
data = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id') ## read data

print(data.head(10))

trafo = PowerTransformer(method='box-cox') ## transformer object

## make column transformer object so it can read the entire column correctly
all_tf = ColumnTransformer(
    [
        ('trafo', trafo, ['Recency', 'Monetary', 'Frequency']), ## Use tarfo object to transform RFM metrics
    ],
    remainder='drop', ## drop non-RFM metrics
    verbose_feature_names_out=False ## shorten feature name by removing transformer's name
)

all_tf.set_output(transform='pandas') ## transform function would return Dataframe

all_tf.fit(data) ## fit data
data_tf = all_tf.transform(data) ## transform input data

print('===')
print(data_tf.head(10))

## save data
if not os.path.exists('data/feat_eng'):
    os.makedirs('data/feat_eng')

data_tf.to_csv('data/feat_eng/data.csv')
sio.dump(all_tf, "data/feat_eng/trf.skops")