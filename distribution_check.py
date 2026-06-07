import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## read data
data = pd.read_csv('data/rfm_in/data.csv', index_col='customer_id') ## read data

## make plotting function
def make_plot(data:pd.DataFrame, col:str):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    if col == 'Frequency':
        sns.histplot(data, x=col, discrete=True, ax=ax)
    else:
        sns.histplot(data, x=col, kde=True, ax=ax)
    ax.set_title(f'{col} Histogram', fontsize=14, fontweight='bold')
    plt.show()

make_plot(data, 'Recency')
make_plot(data, 'Frequency')
make_plot(data, 'Monetary')