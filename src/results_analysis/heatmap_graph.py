import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
datasets = [
    {'file': '../resultados_grid_search/Carbon_Emission/Carbon_Emission_Percentile_Trimming.npy', 'title': 'Carbon Footprint'},
    {'file': '../resultados_grid_search/Wind/Wind_Percentile_Trimming.npy', 'title': 'Wind Speed'},
    {'file': '../resultados_grid_search/House_8L/House_8L_Percentile_Trimming.npy', 'title': 'House_8L'}
]

columns = ['n_estimators', 'group_size', 'max_depth', 'percentile', 'MSE']
n_estimators_limit = 1250  # Set the upper limit for n_estimators

# Set up figure
cols = 3  # Number of columns
rows = 1  # Number of rows
fig, axes = plt.subplots(rows, cols, figsize=(15, 5), constrained_layout=True, gridspec_kw={'wspace': 0.05})

# Iterate through datasets and create heatmaps
for i, dataset in enumerate(datasets):
    # Load the data
    data = np.load(dataset['file'], allow_pickle=True)
    df = pd.DataFrame(data, columns=columns)

    #df = df[df['n_estimators'] <= n_estimators_limit]
    #df = df[df['percentile'] == 2]

    # Sort and pivot data
    df = df.sort_values(by=['group_size', 'percentile'])
    heatmap_data = df.pivot_table(index='group_size', columns='percentile', values='MSE')
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Create a mask for missing values
    mask = heatmap_data.isnull()

    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        ax=axes[i],
        annot=False,
        fmt=".2f",
        cmap="Reds",
        mask=mask,  # Mask for NaN values
        cbar_kws={'label': 'MSE'} if i == cols - 1 else None,  # Only show colorbar on the last heatmap
        linewidths=0.5,  # Adds grid lines
        linecolor='black'  # Grid lines color for NaN regions
    )

    # Fill NaN areas with gray color using the axis patch
    axes[i].patch.set_facecolor('gray')

    # Customize each subplot
    axes[i].set_title(dataset['title'], fontsize=14)
    axes[i].set_xlabel('percentile')
    axes[i].set_ylabel('group_size')

    # Set integer ticks for x-axis and y-axis
    axes[i].set_xticks(range(0, len(heatmap_data.columns), max(1, len(heatmap_data.columns) // 10)))  # Show fewer x-ticks
    axes[i].set_xticklabels(heatmap_data.columns[::max(1, len(heatmap_data.columns) // 10)].astype(int))  # Reduce x-tick labels
    
    axes[i].set_yticks(range(len(heatmap_data.index)))
    axes[i].set_yticklabels(heatmap_data.index.astype(int))

# Adjust layout and show plot
plt.suptitle('MSE Heatmaps (FSC)', fontsize=16, y=1.10)
plt.show()
