##
# =====================================================================================
#
#       Filename:  concatenate_data.py
#
#    Description:  Merge files to get a single csv file
#
#        Version:  1.0
#        Created:  21/03/2023
#       Revision:  14/04/2023
#       Compiler:  python3
#
#         Author:  Idriss Daoudi <idaoudi@anl.gov>
#   Organization:  Argonne National Laboratory
#
# =====================================================================================
##

import pandas as pd
import sys

def mean_of_files():
    dfs = []
    for file in sys.argv[2:]:
        df = pd.read_csv(file, sep='\s*,\s*', engine='python')
        dfs.append(df)
    merged_df = pd.concat(dfs)
    grouped_df = merged_df.groupby(['algorithm', 'matrix_size', 'tile_size', 'case']).mean().reset_index()
    grouped_df.to_csv('energy_data_qr_mean_32768.csv', index=False)

# Counters file first, energy second
def concatenate_files():
    file1 = pd.read_csv(sys.argv[2], skipinitialspace=True)
    file2 = pd.read_csv(sys.argv[3], skipinitialspace=True)
    merged_file = pd.merge(
        file1, file2, on=["algorithm", "matrix_size", "tile_size"])
    merged_file.to_csv("merged_file_cholesky.csv", index=False)

def get_best_cases():
    data = pd.read_csv(sys.argv[2])
    data["total_energy"] = data[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)
    best_cases = data.groupby(['matrix_size', 'tile_size']).apply(
        lambda x: x.loc[
            (x["total_energy"] < x.loc[x["case"] == 1, "total_energy"].values[0]) &
            (x["time"] <= x.loc[x["case"] == 1, "time"].values[0] * 1.05)
        ]
    ).reset_index(drop=True)
    for matrix_size in best_cases['matrix_size'].unique():
        for tile_size in best_cases['tile_size'].unique():
            print(f"Matrix Size: {matrix_size}, Tile Size: {tile_size}")
            matrix_tile_best_cases = best_cases[(best_cases['matrix_size'] == matrix_size) & (best_cases['tile_size'] == tile_size)]
            cases = ' '.join(str(case) for case in matrix_tile_best_cases['case'].unique())
            print(cases)
            print()

def merge_files():
    file1 = pd.read_csv(sys.argv[2]) # counters
    file2 = pd.read_csv(sys.argv[3]) # energy and time

    file1.columns = [col.strip() for col in file1.columns]

    task_mapping = {task: f"task{i+1}" for i, task in enumerate(file1['task'].unique())}

    file1['task'] = file1['task'].map(task_mapping)

    file1_pivot = file1.pivot_table(
        index=['algorithm', 'matrix_size', 'tile_size'],
        columns='task',
        values=['mem_boundness', 'arithm_intensity', 'bmr', 'ilp', 'l3_cache_ratio']
    ).reset_index()

    file1_pivot.columns = [
        '_'.join(col).strip().replace(' ', '') if col[1] else col[0]
        for col in file1_pivot.columns.values
    ]

    merged = pd.merge(file2, file1_pivot, on=['algorithm', 'matrix_size', 'tile_size'])
    merged.to_csv('merged_qr_16384.csv', index=False)

if __name__ == '__main__':
    key = int(sys.argv[1])

    if (key == 1):
        ret = mean_of_files()
    if (key == 2):
        ret = concatenate_files()
    if (key == 3):
        get_best_cases()
    if (key == 4):
        merge_files()