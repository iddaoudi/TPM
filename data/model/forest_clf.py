import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

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

data_train = pd.read_csv(sys.argv[1])
data_test = pd.read_csv(sys.argv[2])

data_train['Energy'] = data_train['PKG1'] + data_train['PKG2'] + data_train['DRAM1'] + data_train['DRAM2']
data_test['Energy'] = data_test['PKG1'] + data_test['PKG2'] + data_test['DRAM1'] + data_test['DRAM2']

cols_to_sum = ['task{}_{}'.format(i, metric)
               for i in range(1, 5)
               for metric in ('mem_boundness', 'arithm_intensity', 'ilp', 'l3_cache_ratio')]

data_train['sum'] = data_train.loc[:, cols_to_sum].sum(axis=1)
data_test['sum'] = data_test.loc[:, cols_to_sum].sum(axis=1)

data_train['bool'] = data_train.apply(lambda row: conditions_fulfilled(row, data_train), axis=1)
data_test['bool'] = data_test.apply(lambda row: conditions_fulfilled(row, data_test), axis=1)

features = ['tile_size', 'sum', 'task1', 'task2', 'task3', 'task4']
target = 'bool'

X_train = data_train[features]
y_train = data_train[target]

X_test = data_test[features]
y_test = data_test[target]

clf = RandomForestClassifier(n_estimators=100, random_state=1, class_weight=None)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# After training the model
importances = clf.feature_importances_
feature_list = list(features)

for feature, importance in zip(feature_list, importances):
    print(f"The importance of feature {feature} is: {importance}")


            # 'task4_mem_boundness','task4_arithm_intensity','task4_ilp','task4_l3_cache_ratio',
            # 'task2_mem_boundness','task2_arithm_intensity','task2_ilp','task2_l3_cache_ratio',
            # 'task3_mem_boundness','task3_arithm_intensity','task3_ilp','task3_l3_cache_ratio',
            # 'task1_mem_boundness','task1_arithm_intensity','task1_ilp','task1_l3_cache_ratio']
# print(y_pred)
# print(y_test)