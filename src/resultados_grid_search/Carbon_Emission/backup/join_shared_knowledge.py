import numpy as np
import pandas as pd

# Define the file paths
file_paths = ['Carbon_Emission_Shared_Knowledge_0.npy',
              'Carbon_Emission_Shared_Knowledge_1.npy',
              'Carbon_Emission_Shared_Knowledge_2.npy',
              'Carbon_Emission_Shared_Knowledge_3.npy',
              'Carbon_Emission_Shared_Knowledge_4.npy',
              'Carbon_Emission_Shared_Knowledge_5.npy']

# Define the column names
columns = ['n_estimators', 'group_size', 'max_depth', 'initial_max_depth', 'MSE']

# Initialize an empty list to store the DataFrames
dfs = []

# Loop through the file paths and load the data
for file_path in file_paths:
    data_carbon = np.load(file_path)
    df_carbon = pd.DataFrame(data_carbon, columns=columns)
    dfs.append(df_carbon)

# Concatenate the DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

np.save('Carbon_Emission_Shared_Knowledge.npy', merged_df.to_numpy())