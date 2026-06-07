import pandas as pd
import os
import datetime as dt

## read data
data = pd.read_csv('data/source/data.csv', parse_dates=['order_date'],
                   date_format={'order_date':'%Y-%m-%d'}) 

print(data.head(10))
print('===')

## this data has no NaN
kosong = data[data.isna().any(axis=1)]
print(f'Number of rows with missing value: {len(kosong)}')

## check if there is duplicated data on orders
unika = data[data['order_id'].duplicated()]
print(f'Number of duplicated order id: {len(unika)}')

## check if there is duplicated order-customer pairs
unikb = data[data[['customer_id','order_id']].duplicated()]
print(f'Number of duplicated customer-order pairs: {len(unikb)}')

# ## check if order date is in between 2024-2025
# oldest = dt.date(year=2024, month=1, day=1)
# recent = dt.date(year=2025, month=12, day=31)
# wrong_date = data[(data['order_date'].dt.date<oldest) | (data['order_date'].dt.date>recent)]
# print(f'Number of wrong date range: {len(wrong_date)}')

## ensure that there are no negative spending
neg_spend = data[(data['total_spend']<0)]
print(f'Number of negative spending: {len(neg_spend)}')

## we have order_date and order_time, we attempt to combine that
data['order_time'] = data['order_date'].astype(str) + " " + data['order_time']
data.drop('order_date', axis=1, inplace=True)
data['order_time'] = pd.to_datetime(data['order_time'])

## translate day_of_week as weekend (0) or not (1)
kamus = {'Mon': False, 'Tue': False, 'Wed': False, 'Thu': False, 'Fri': False, 'Sat': True, 'Sun': True}
data['is_weekend'] = data['day_of_week'].replace(kamus)

## extract date-time data on order_time, here we simply focus on hour
data['order_hour'] = data['order_time'].dt.hour

## translate hour data to morning, afternoon, evening, night
def time_seg(val):
    if val >= 5 and val < 12:
        return 'morning'
    if val >= 12 and val < 17:
        return 'afternoon'
    if val >= 17 and val < 21:
        return 'evening'
    else:
        return 'night'
data['hour_segment'] = data['order_hour'].apply(time_seg)

# print(data.head(10))

if not os.path.exists('data/prep'):
    os.makedirs('data/prep')

data.to_csv('data/prep/data.csv', index=False)
