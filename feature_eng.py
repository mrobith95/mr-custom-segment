import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
import skops.io as sio

## read data
data = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id') ## read data

print(data.head(10))

trafo = PowerTransformer(method='box-cox') ## transformer object

# ## prepare transformations
# time_tf = OneHotEncoder(drop='if_binary', ## drop 1 column if binary
#                         sparse_output=False, ## let the data dense for easier reading
#                         handle_unknown='ignore' ## set unknown category to all 0
#                         )
# mr_tf   = QuantileTransformer(output_distribution='normal', ## transform data to normal distribution
#                               random_state=300 ## enforce replication
#                               )
# fsc_tf  = PowerTransformer() ## method is yeo-johnson, for any real numbers

## unite all tarnsformer
all_tf = ColumnTransformer(
    [
        # ('time_tf', time_tf, ['Weekend', 'Time']), ## Time related data transformed by OneHot
        ('trafo', trafo, ['Recency', 'Monetary', 'Frequency']), ## Use tarfo object to transform RFM metrics
        # ('fsc_tf', fsc_tf, ['Frequency']) ## Frequency, Size, Custom by yeo-johnson
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