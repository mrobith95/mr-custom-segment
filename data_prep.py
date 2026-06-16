import pandas as pd
import os

## read data
data = pd.read_csv('data/source/data.csv') ## read data

print(data.head(10))
print('===')

## this data has no NaN

## we have order_date and order_time, we attempt to combine that
data['order_time'] = data['order_date'] + " " + data['order_time']
data.drop('order_date', axis=1, inplace=True)
data['order_time'] = pd.to_datetime(data['order_time'])

## translate day_of_week as weekend (0) or not (1)
kamus = {'Mon': False, 'Tue': False, 'Wed': False, 'Thu': False, 'Fri': False, 'Sat': True, 'Sun': True}
data['is_weekend'] = data['day_of_week'].replace(kamus)

## extract date-time data on order_time, here we simply focus on hour
data['order_hour'] = data['order_time'].dt.hour

# ## translate hour data to morning, afternoon, evening, night
def time_seg(val):
    if val >= 5 and val < 12:
        return 'morning'
    if val >= 12 and val < 17:
        return 'afternoon'
    # if val >= 17 and val < 21:
    #     return 'evening'
    else:
        return 'night'
data['hour_segment'] = data['order_hour'].apply(time_seg)

print(data.head(10))

if not os.path.exists('data/prep'):
    os.makedirs('data/prep')

data.to_csv('data/prep/data.csv', index=False)
