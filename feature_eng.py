import pandas as pd
import os
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import skops.io as sio

## read data
data = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id') ## read data

print(data.head(10))

## prepare transformations. no categoricals
num_tf  = Pipeline(
    [
        ('yeo-johnson', PowerTransformer(standardize=True)), ## method is yeo-johnson, for any real numbers, no need to standardize
    ]
)

ord_tf = Pipeline(
    [
        ## Ordinal transform first
        ('ordinal', 
         OrdinalEncoder(categories=[['18-24','25-34','35-44','45-54','55+']],
                        handle_unknown='use_encoded_value', ## if unknown category is used ...
                        unknown_value=-1, ## use -1
                        encoded_missing_value=-1 ## NaN also use -1
            )
        )
    ]
)


## unite all tarnsformer
all_tf = ColumnTransformer(
    [
        ## Numerical features are scaled by yeo-johson and scaled so max dist = 1
        ('num_tf', num_tf, ['Recency','Frequency','Monetary','Size','Custom Rate','Waiting Time','Satisfaction']),
        ## Ordinal
        ('ord_tf', ord_tf, ['Age Group'])
    ],
    remainder='passthrough', ## let non-transformed feature pass
    verbose_feature_names_out=False ## shorten feature name by removing transformer's name
)

all_tf.set_output(transform='pandas') ## transform function would return Dataframe

# print(data.info())
all_tf.fit(data) ## fit data
data_tf = all_tf.transform(data) ## transform input data

## make function that deal with ordinals
def ordinal_filter(val, ref=0):
    if val>ref:
        return True
    else:
        return False
    
data_tf['Age Group>18-24'] = data_tf['Age Group'].apply(ordinal_filter, ref=0)
data_tf['Age Group>25-34'] = data_tf['Age Group'].apply(ordinal_filter, ref=1)
data_tf['Age Group>35-44'] = data_tf['Age Group'].apply(ordinal_filter, ref=2)
data_tf['Age Group>44-54'] = data_tf['Age Group'].apply(ordinal_filter, ref=3)

## make function that deal with cyclic categoricals
def cyc_filter(val, refs):
    if val in refs:
        return True
    else:
        return False
    
data_tf['Time_0'] = data_tf['Time'].apply(cyc_filter, refs=['afternoon', 'evening'])
data_tf['Time_1'] = data_tf['Time'].apply(cyc_filter, refs=['evening', 'night'])

data_tf.drop(['Age Group', 'Time'], axis=1, inplace=True)

print(data_tf.info())

print('===')

## save data
if not os.path.exists('data/feat_eng'):
    os.makedirs('data/feat_eng')

data_tf.to_csv('data/feat_eng/data.csv')
sio.dump(all_tf, "data/feat_eng/trf.skops")