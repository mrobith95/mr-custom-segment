import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pickle import dump, load
import os

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    'likithagedipudi/starbucks-customer-ordering-patterns',
    'starbucks_customer_ordering_patterns.csv'
)

## saving data
if not os.path.exists('data/raw'):
    os.makedirs('data/raw')
if not os.path.exists('data/source'):
    os.makedirs('data/source')

df.to_csv('data/raw/raw_data.csv')

df = df.drop_duplicates() ## drop duplicates

df.to_csv('data/source/data.csv', index=False)