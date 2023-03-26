##
# =====================================================================================
#
#       Filename:  concatenate_data.py
#
#    Description:  Merge files to get a single csv file
#
#        Version:  1.0
#        Created:  21/03/2023
#       Revision:  23/03/2023
#       Compiler:  python3
#
#         Author:  Idriss Daoudi <idaoudi@anl.gov>
#   Organization:  Argonne National Laboratory
#
# =====================================================================================
##

import pandas as pd
import sys

# Counters file first, energy second
def concatenate_files():
    file1 = pd.read_csv(sys.argv[1], skipinitialspace=True)
    file2 = pd.read_csv(sys.argv[2], skipinitialspace=True)
    merged_file = pd.merge(
        file1, file2, on=["algorithm", "matrix_size", "tile_size"])
    merged_file.to_csv("merged_file_cholesky.csv", index=False)
    return 0

def mean_of_files():
    # Read the CSV files into separate dataframes
    dfs = []
    for file in sys.argv[1:]:
        df = pd.read_csv(file, sep='\s*,\s*', engine='python')
        dfs.append(df)
    # Concatenate the dataframes vertically
    merged_df = pd.concat(dfs)
    # Group the data by algorithm, matrix_size, tile_size, and case, and calculate the mean of PKG1, PKG2, DRAM1, DRAM2, and time
    grouped_df = merged_df.groupby(['algorithm', 'matrix_size', 'tile_size', 'case']).mean().reset_index()
    # Write the result to a new CSV file
    grouped_df.to_csv('energy_data_cholesky_mean.csv', index=False)

    return 0

def get_best_cases():
    data = pd.read_csv(sys.argv[1])
    data["total_energy"] = data[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)
    best_cases = data.groupby(['matrix_size', 'tile_size']).apply(
        lambda x: x.loc[
            (x["total_energy"] < x.loc[x["case"] == 1, "total_energy"].values[0]) &
            (x["time"] <= x.loc[x["case"] == 1, "time"].values[0] * 1.15)
        ]
    ).reset_index(drop=True)
    for matrix_size in best_cases['matrix_size'].unique():
        for tile_size in best_cases['tile_size'].unique():
            print(f"Matrix Size: {matrix_size}, Tile Size: {tile_size}")
            matrix_tile_best_cases = best_cases[(best_cases['matrix_size'] == matrix_size) & (best_cases['tile_size'] == tile_size)]
            cases = ' '.join(str(case) for case in matrix_tile_best_cases['case'].unique())
            print(cases)
            print()

if __name__ == '__main__':
    mean = 0
    concatenate = 0
    best_cases = 1

    if (mean):
        ret = mean_of_files()
        assert (ret == 0)
    if (concatenate):
        ret = concatenate_files()
        #assert (ret == 0)
    if (best_cases):
        get_best_cases()