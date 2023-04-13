import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load the data
data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])

# Compute energy and energy efficiency
data_train['Energy'] = data_train['PKG1'] + data_train['PKG2'] + data_train['DRAM1'] + data_train['DRAM2']
data_test['Energy'] = data_test['PKG1'] + data_test['PKG2'] + data_test['DRAM1'] + data_test['DRAM2']

# Prepare the features and target for training and testing
features = ['ilp', 'mem_boundness', 'bmr', 'arithm_intensity', 'l3_cache_ratio', 'matrix_size', 'tile_size']
target = 'Energy'

X_train = data_train[features]
y_train = data_train[target]

X_test = data_test[features]
y_test = data_test[target]

# Train a random forest regressor
regr = RandomForestRegressor(n_estimators=100, random_state=42)
regr.fit(X_train, y_train)

# Get feature importances (weights)
importances = regr.feature_importances_

# Calculate the combined metric for test data
data_train['combined_metric'] = np.dot(X_train, importances)
data_test['combined_metric'] = np.dot(X_test, importances)

# Calculate the product of energy and time for the plots
data_train['Energy_Time'] = data_train['Energy'] * data_train['time']
data_test['Energy_Time'] = data_test['Energy'] * data_test['time']

def conditions_fulfilled(row, data):
    case_1_data = data.loc[(data['task'] == row['task']) & (data['matrix_size'] == row['matrix_size']) & (data['tile_size'] == row['tile_size']) & (data['case'] == 1)]
    
    if not case_1_data.empty:
        case_1_energy = case_1_data['Energy'].mean()
        case_1_time = case_1_data['time'].mean()
        
        condition1 = row['Energy'] <= case_1_energy
        condition2 = row['time'] <= case_1_time * 1.05
        return condition1 and condition2
    else:
        return False

# Apply the function to the training data
data_train['conditions_fulfilled'] = data_train.apply(lambda row: conditions_fulfilled(row, data_train), axis=1)
data_test['conditions_fulfilled'] = data_test.apply(lambda row: conditions_fulfilled(row, data_test), axis=1)

def plot_with_markers(x, y, hue, style, text, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    sns.scatterplot(x=x, y=y, hue=hue, style=style, data=data, **kwargs)
    for i, txt in enumerate(data[text]):
        ax.annotate(txt, (data[x].iloc[i], data[y].iloc[i]), fontsize=8)

# Create a FacetGrid plot with a separate plot for each task
g = sns.FacetGrid(data=data_train, col='task', height=4, aspect=1, legend_out=True)

# Map the scatterplot to the FacetGrid with different markers for true and false cases
g.map_dataframe(plot_with_markers, x='combined_metric', y='Energy_Time', hue='conditions_fulfilled', style='conditions_fulfilled', text='case', alpha=.7, edgecolor='w', markers=['o', 'X'], palette=['red', 'green'])

# Configure the plot
g.set_axis_labels('Combined Metric', 'Energy * Time')
g.add_legend(title='Conditions Fulfilled')

plt.tight_layout()
plt.savefig("cholesky_16384.png")
plt.show()

g = sns.FacetGrid(data=data_test, col='task', height=4, aspect=1, legend_out=True)

# Map the scatterplot to the FacetGrid with different markers for true and false cases
g.map_dataframe(plot_with_markers, x='combined_metric', y='Energy_Time', hue='conditions_fulfilled', style='conditions_fulfilled', text='case', alpha=.7, edgecolor='w', markers=['o', 'X'], palette=['red', 'green'])

# Configure the plot
g.set_axis_labels('Combined Metric', 'Energy * Time')
g.add_legend(title='Conditions Fulfilled', bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.savefig("cholesky_32768.png")
plt.show()

