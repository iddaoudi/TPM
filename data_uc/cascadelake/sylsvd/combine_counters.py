import pandas as pd
import glob
import os
import sys
import re

def combine_csv_files(folder_name):
    # Get a list of all the csv files in the folder
    files = glob.glob(f"{folder_name}/counters_{folder_name}_*_*[.csv]")

    # Extract the unique base filenames by removing the last "_{number}.csv" part
    base_filenames = set(os.path.splitext(filename)[0].rsplit('_', 1)[0] for filename in files)
    
    for base_filename in base_filenames:
        # Construct a pattern for all files with the same base filename
        file_pattern = base_filename + "_*.csv"
        matched_files = glob.glob(file_pattern)
        
        # Ensure files are sorted in order to get the correct order of columns in final csv
        matched_files.sort()
        
        df_list = []
        
        # Loop through all the matched files
        for filename in matched_files:
            # Read each file into a pandas DataFrame
            df = pd.read_csv(filename, index_col=False)
            
            # Drop any columns that have all NaN values (these would be the 'Unnamed' columns)
            df = df.dropna(how='all', axis=1)
            
            # Add the DataFrame to the list
            df_list.append(df)
        
        # Concatenate all the DataFrames in the list
        df_concat = pd.concat(df_list, axis=1)
        
        # Remove duplicate columns in the dataframe
        df_concat = df_concat.loc[:,~df_concat.columns.duplicated()]
        
        # Write the result to a new csv file
        output_filename = base_filename + ".csv"
        df_concat.to_csv(output_filename, index=False)

if __name__ == "__main__":
    folder_name = sys.argv[1].strip('/')
    combine_csv_files(folder_name)
