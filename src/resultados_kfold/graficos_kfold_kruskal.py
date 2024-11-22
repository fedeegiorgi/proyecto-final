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
    {'file': 'score/score_Flight.csv', 'title': 'Flight Price'},
    {'file': 'score/score_Rainfall.csv', 'title': 'Rainfall'},
    {'file': 'score/score_Abalone.csv', 'title': 'Abalone'}
]

# Define the grid size
num_datasets = len(datasets)
cols = 3  
rows = (num_datasets + cols - 1) // cols  # Calculate rows based on dataset count and columns

# Define consistent order for models
model_order = ['RF', 'SK', 'OOB', 'IQR', 'OOB+IQR', 'PERT', 'FSC']

# Create the figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = axes.flatten()  

# Ensure save directory exists
output_dir = "graficos_kfold"
os.makedirs(output_dir, exist_ok=True)

# Iterate through datasets and plot
for i, dataset in enumerate(datasets):
    # Check if file exists
    if not os.path.exists(dataset['file']):
        print(f"Warning: File {dataset['file']} not found. Skipping.")
        continue

    # Load dataset
    data = pd.read_csv(dataset['file'])

    # Adjust MSE scaling based on the dataset title
    scale_factors = {
        'Carbon Footprint': 100000,
        'Wind Speed': 10,
        'House_8L': 1000,
        'Flight Price': 1,
        'Rainfall': 1,
        'Abalone': 1
    }
    if dataset['title'] in scale_factors:
        data['MSE'] /= scale_factors[dataset['title']]

    # Map old model names to new ones
    rename_mapping = {
        'RandomForestRegressor': 'RF',
        'SharedKnowledgeRandomForestRegressor': 'SK',
        'OOBRandomForestRegressor': 'OOB',
        'IQRRandomForestRegressor': 'IQR',
        'OOBPlusIQRRandomForestRegressor': 'OOB+IQR',
        'OOB_plus_IQR': 'OOB+IQR',
        'PercentileTrimmingRandomForestRegressor': 'PERT',
        'FirstSplitCombinerRandomForestRegressor': 'FSC',
        'RFRegressorFirstSplitCombiner': 'FSC'
    }
    data['Model'] = data['Model'].map(rename_mapping)

    # Calculate the median MSE for each unique model
    median_mse = data.groupby('Model')['MSE'].median()

    # Reindex to ensure consistent order across all datasets
    median_mse = median_mse.reindex(model_order)

    # Use the Viridis colormap
    num_bars = len(median_mse)
    viridis = plt.get_cmap('viridis')  # Compatible with older Matplotlib versions
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
    fig.delaxes(axes[j])

# Save the plot
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust spacing between plots
output_path = os.path.join(output_dir, "comparison_grid.png")
plt.savefig(output_path, dpi=300)  # Save with high resolution
plt.show()
