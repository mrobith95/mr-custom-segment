# mr-custom-segment
python scripts for my Medium Article on Customer Segmentation: -

## Background
Customers are the most vital part of a business, regardless of its sector. Without them, any business would collapse due to loss of revenue. Thus, businesses try to keep their customers at all times, while also attempting to acquire as many customers as possible. One-way businesses can use to keep or acquire customers is by studying their behavior. This study could later be used by related departments to develop effective action. For example, marketing could use customer behavior study to make personalized marketing campaign. One well-known analysis that can be derived from this order information is Recency-Frequency-Monetary Analysis (RFM Analysis), in which we solve using K-Means Clustering.

## Quick Notes
* Dataset: [Starbucks Customer Ordering Patterns](https://www.kaggle.com/datasets/likithagedipudi/starbucks-customer-ordering-patterns)
* Model: K-Means
* Packages: Available on requirements.txt

## Intended Pipeline
Run the following codes in sequence.
1. download_data.py: Download data from kaggle and remove duplicates.
2. data_prep.py: Fixing NaNs, typos, and possible invalid entries.
3. build_rfm_input.py: Compute RFM metrics for each customer.
4. distribution_check.py: Plot RFM's historgram to select suitable preprocessing.
5. feature_eng.py: Preprocess RFM metrics before fed into model.
6. modelling.py: Perform Kmeans with several number of clusters.
7. analyze_cluster.py: Plot metrics' boxplot for each cluster.
