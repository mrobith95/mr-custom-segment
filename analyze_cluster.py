import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid") ## set sns layout and styling

## read data for modelling and for rfm input
data = pd.read_csv('data/modelling/data.csv', index_col='customer_id') ## read data

## helper function for aggregated mode and ratio
def get_mode(series):
    modes = series.mode()
    return modes.iloc[0] if not modes.empty else np.nan

def get_ratio(series):
    hitung = len(series)
    return hitung/len(data)

## function for plotting distirbution of clustered data
def plot_cluster_distributions(df, features_to_plot, cluster_column='cluster'):
    """
    Plots the distribution of specific features across clusters, showing 
    percentages for categorical/binary columns.
    
    Parameters:
    df (pd.DataFrame): The original, UNSCALED raw data with the cluster labels attached.
    cluster_column (str): The name of the column containing the cluster labels.
    features_to_plot (list or str): A single feature name or list of features to inspect.
                                    If None, defaults to all features.
    """
        
    # Verify features actually exist in the dataframe
    if features_to_plot not in df.columns:
        print("None of the specified features were found in the DataFrame.")
        return

    # Define underlying data types for proper plot routing
    numeric_types = ['Recency', 'Frequency', 'Monetary', 'Size', 'Custom Rate', 'Waiting Time',
                     'Satisfaction']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # # Ensure axes is a flat array even if it's a single plot or a single row
    # ax = [ax]
    
        # --- PATHWAY A: NUMERIC FEATURES (Boxplots) ---
    if features_to_plot in numeric_types:
        sns.boxplot(
            ax=ax,
            data=df,
            y=cluster_column,
            x=features_to_plot,
            palette="Set2",
            hue=cluster_column,
            legend=False,
            orient='h'
        )
        ax.set_title(f'{features_to_plot} Distribution by Cluster', fontsize=14, fontweight='bold')
        ax.set_xlabel(features_to_plot, fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        
        # Dynamic log scale for highly skewed columns
        if features_to_plot in ['Monetary', 'Frequency']:
            ax.set_xscale('log')
            ax.set_xlabel(f'{features_to_plot} (Log Scale)', fontsize=12)
            
    # --- PATHWAY B: CATEGORICAL FEATURES (Normalized Percentage Barplots) ---
    else:
        # Group by cluster and the category, calculate size, then normalize by cluster size
        percentage_df = (
            df.groupby(cluster_column)[features_to_plot]
            .value_counts(normalize=True)
            .rename('percentage')
            .reset_index()
        )
        # Convert decimal to 0-100% scale
        percentage_df['percentage'] = percentage_df['percentage'] * 100
        
        sns.barplot(
            ax=ax,
            data=percentage_df,
            y=cluster_column,
            x='percentage',
            orient='h',
            hue=features_to_plot,
            palette="muted"
        )
        ax.set_title(f'{features_to_plot} Composition (%) by Cluster', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_xlim(0, 105) # Keep headroom for legends
        
        # Add "%" labels to the y-axis ticks
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))
        ax.legend(title=features_to_plot, loc='lower right')

    plt.tight_layout()
    plt.show()

print(data.head(10))

## compute percentage of data for each cluster
group_data = data.groupby('cluster')

agg_data = (
    group_data
    .agg(
        {
            'cluster': get_ratio
        }
    )
    .rename(
        columns={
            'cluster': 'Count'
        }
    )
)

print(agg_data)

print(data.columns)
for col in data.columns:
    if col != 'cluster':
        plot_cluster_distributions(data,
                                   features_to_plot=col)
