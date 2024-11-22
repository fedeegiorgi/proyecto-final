import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

# Dataset list
datasets = [
    {'file': 'score/score_Carbon_Emission.csv', 'title': 'Carbon Footprint'},
    {'file': 'score/score_Wind.csv', 'title': 'Wind Speed'},
    {'file': 'score/score_House_8L.csv', 'title': 'House_8L'},
]

# Define the grid size
num_datasets = len(datasets)
cols = 3  # Number of columns in the grid
rows = (num_datasets + cols - 1) // cols  # Calculate rows needed

# Define consistent order for models
model_order = ['RF', 'SK', 'OOB', 'IQR', 'OOB + IQR', 'PERT', 'FSC']

# Create the figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = axes.flatten()  # Flatten to iterate easily

# Iterate through datasets and plot
for i, dataset in enumerate(datasets):
    # Load dataset
    data = pd.read_csv(dataset['file'])

    if dataset['title'] == 'Carbon Footprint':
        data['MSE'] = data['MSE'] / 100000
    elif dataset['title'] == 'Wind Speed':
        data['MSE'] = data['MSE'] / 10
    elif dataset['title'] == 'House_8L':
        data['MSE'] = data['MSE'] / 1000

    # Calculate the median MSE for each model
    median_mse = data.groupby('Model')['MSE'].median()

    # Map old model names to new ones
    rename_mapping = {
        'RandomForestRegressor': 'RF',
        'SharedKnowledgeRandomForestRegressor': 'SK',
        'OOBRandomForestRegressor': 'OOB',
        'IQRRandomForestRegressor': 'IQR',
        'OOB_plus_IQR': 'OOB + IQR',
        'PercentileTrimmingRandomForestRegressor': 'PERT',
        'RFRegressorFirstSplitCombiner': 'FSC',
    }
    median_mse.index = median_mse.index.map(rename_mapping)

    # Reindex to ensure consistent order across all datasets
    median_mse = median_mse.reindex(model_order)

    # Use the Viridis colormap
    num_bars = len(median_mse)
    viridis = cm.get_cmap('viridis', num_bars)
    colors = viridis(np.linspace(0, 1, num_bars))

    # Plot bar graph
    ax = axes[i]
    median_mse.plot(kind='bar', ax=ax, color=colors, edgecolor='black')

    # Customize
    ax.set_title(dataset['title'], fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Median MSE', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

# Remove any unused axes if the grid is larger than the dataset count
for j in range(num_datasets, len(axes)):
    fig.delaxes(axes[j])  # Remove empty subplots

# Guardar el gráfico
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Espacio vertical (hspace) y horizontal (wspace)
output_path = os.path.join("graficos_kfold", "comparison_grid.png")
plt.savefig(output_path, dpi=300)  # Guardar con alta resolución
plt.show()