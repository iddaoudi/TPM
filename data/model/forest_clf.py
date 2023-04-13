import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])

data_train['Energy'] = data_train['PKG1'] + data_train['PKG2'] + data_train['DRAM1'] + data_train['DRAM2']
data_test['Energy'] = data_test['PKG1'] + data_test['PKG2'] + data_test['DRAM1'] + data_test['DRAM2']

def conditions_fulfilled(row, data):
    case_1_data = data.loc[(data['algorithm'] == row['algorithm']) & (data['matrix_size'] == row['matrix_size']) & (data['tile_size'] == row['tile_size']) & (data['case'] == 1)]    
    if not case_1_data.empty:
        case_1_energy = case_1_data['Energy'].mean()
        case_1_time = case_1_data['time'].mean()
        
        condition1 = row['Energy'] <= case_1_energy
        condition2 = row['time'] <= case_1_time * 1.05
        return condition1 and condition2
    else:
        return False

data_train['bool'] = data_train.apply(lambda row: conditions_fulfilled(row, data_train), axis=1)
data_test['bool'] = data_test.apply(lambda row: conditions_fulfilled(row, data_test), axis=1)

features = ['matrix_size', 'tile_size', 
            'arithm_intensity_task1', 'arithm_intensity_task2', 'arithm_intensity_task3', 'arithm_intensity_task4',
            'bmr_task1', 'bmr_task2', 'bmr_task3', 'bmr_task4',
            'ilp_task1', 'ilp_task2', 'ilp_task3', 'ilp_task4',
            'l3_cache_ratio_task1', 'l3_cache_ratio_task2', 'l3_cache_ratio_task3', 'l3_cache_ratio_task4',
            'mem_boundness_task1', 'mem_boundness_task2', 'mem_boundness_task3', 'mem_boundness_task4']
target = 'bool'

X_train = data_train[features]
y_train = data_train[target]

X_test = data_test[features]
y_test = data_test[target]

clf = RandomForestClassifier(n_estimators=100, random_state=0, class_weight={True: 1, False: 0})
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
