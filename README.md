# mr-custom-segment
python scripts for my Medium Article on Customer Segmentation: [Enhanced RFM Analysis for Customer Segmentation using K-Prototypes]()

## Background
In the [previous article](https://medium.com/@mrobith95/rfm-analysis-for-customer-segmentation-using-k-means-clustering-coffee-shop-case-study-7ed5a3dd654e), we performed basic RFM Analysis using K-Means. However, customer behaviour is not limited to spending activity alone. Adding this behaviour to our analysis would help us determine not only how customers behave in their spending but also their actual ordering behaviour. The main challenge of this analysis is that some customer behaviours are not numerical. Thus, K-Means that we use for basic RFM analysis can't work since it only accepts numerical values. This article will use K-Prototypes to solve this clustering problem.

## Quick Notes
* Dataset: [Starbucks Customer Ordering Patterns](https://www.kaggle.com/datasets/likithagedipudi/starbucks-customer-ordering-patterns)
* Model: K-Prototypes
* Packages: Available on requirements.txt

## Intended Pipeline
Run the following codes in sequence.
1. download_data.py: Download data from kaggle and remove duplicates.
2. data_prep.py: Fixing NaNs, typos, and possible invalid entries.
3. build_rfm_input.py: Compute RFM metrics, behaviour and personaility features for each customer.
4. feature_eng.py: Preprocess RFM metrics, behaviour and personaility features before fed into model.
5. modelling.py: Perform Kprototypes with several number of clusters.
6. analyze_cluster.py: Plot metrics' boxplot for each cluster.

## variation
Visit the [main branch](https://github.com/mrobith95/mr-custom-segment/tree/main) for classic RFM Analysis for Customer Segmentation using K-Means
