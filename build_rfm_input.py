import pandas as pd
import os
import numpy as np

# Safe mode helper function to handle ties on mode cleanly
def get_mode(series):
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else np.nan

## read data
data = pd.read_csv('data/prep/data.csv', parse_dates=['order_time']) ## read data

print(data.info())

## simulate the earliest day the data can be processed
snapshot_date = data["order_time"].max() + pd.Timedelta(days=1)

rfm = (
    data.groupby("customer_id")
    .agg(
        {   # define RFM metric
            # Recency: Days since last order
            "order_time": lambda x: (snapshot_date - x.max()).days,
            # Frequency: Total number of orders
            "customer_id": "count",
            # Monetary: Total spending
            "total_spend": "sum",
            # define grouping metric
            # Is Weekend: Is customer tend to order at weekend or not
            "is_weekend": get_mode,
            # Time Segment: At what time segment do customer order
            "hour_segment": get_mode,
            # Channel: Favorite ordering channel
            "order_channel": get_mode,
            # location: favorite store location
            "store_location_type": get_mode,
            # region: favorite store region
            "region": get_mode,
            # age group: 
            "customer_age_group": get_mode,
            # gender
            "customer_gender": get_mode,
            # Size: How many items customer tend to order each order
            "cart_size": 'median', ## good for determine typical behaviour
            # Custom Rate: Average of customization they order
            "num_customizations": 'mean',
            # waiting time:
            'fulfillment_time_min': 'median',
            # drink: favorite drink
            'drink_category': get_mode,
            # food: if they also order food
            'has_food_item': get_mode,
            # order_ahead: do they prefer ordering ahead of time (mobile only)
            'order_ahead': get_mode,
            # satisfaction: mean of customer satisfaction
            'customer_satisfaction': 'mean'

        }
    )
    .rename(
        columns={
            "order_time": "Recency",
            "customer_id": "Frequency",
            "total_spend": "Monetary",
            "is_weekend": 'Weekend',
            "hour_segment": 'Time',
            "order_channel": "Channel",
            'store_location_type': 'Location',
            'customer_age_group': 'Age Group',
            'customer_gender': 'Gender',
            "cart_size": "Size",
            "num_customizations": "Custom Rate",
            'fulfillment_time_min': 'Waiting Time',
            'drink_category': 'Drink',
            'has_food_item': 'Food',
            'order_ahead': 'Order Ahead',
            'customer_satisfaction': 'Satisfaction'

        }
    )
)

rfm.sort_index(inplace=True)

print(rfm.head(10))

## save data
if not os.path.exists('data/rfm_in'):
    os.makedirs('data/rfm_in')

rfm.to_csv('data/rfm_in/data.csv')