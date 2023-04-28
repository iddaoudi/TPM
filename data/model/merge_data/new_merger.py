import sys
import pandas as pd
import dictionaries

metrics = ['mem_boundness','arithm_intensity','ilp','l3_cache_ratio']

def read_file():
    df_energy_and_time   = pd.read_csv(sys.argv[1])
    df_counters_original = pd.read_csv(sys.argv[2])
    df_counters_reduced  = pd.read_csv(sys.argv[3])
    return df_energy_and_time, df_counters_original, df_counters_reduced

def extract_tasks(df_counters):
    task_list = list({task for task in df_counters['task']})
    task_dict = {elem: f"task{idx+1}" for idx, elem in enumerate(task_list)}
    return task_list, task_dict

def logic(df_energy_and_time, df_counters_original, df_counters_reduced, task_dict):
    unique_algorithm = df_energy_and_time['algorithm'].unique()
    unique_matrix    = df_energy_and_time['matrix_size'].unique()
    unique_tile      = df_energy_and_time['tile_size'].unique()
    unique_cases     = df_energy_and_time['case'].unique()
    
    if unique_algorithm == 'cholesky':
        dico = dictionaries.cholesky_dict
        
    for task in dico[16]:
        for metric in metrics:
            column_name = task_dict[task] + "_" + metric
            if column_name not in df_energy_and_time.columns:
                df_energy_and_time = df_energy_and_time.assign(**{column_name: 0}) 

    for i in range(len(metrics)):
        for algorithm in unique_algorithm:
            for matrix in unique_matrix:
                for tile in unique_tile:
                    for case in unique_cases:
                        reduced_frequency_tasks = dico[case]
                        original_frequency_tasks = []
                        # First populate the original frequency list
                        for t in dico[16]:
                            if t not in reduced_frequency_tasks:
                                original_frequency_tasks.append(t)
                        # Then do the logic
                        # Reduced frequency tasks first
                        for task in reduced_frequency_tasks:
                            column_name = task_dict[task] + "_" + metrics[i]
                            
                            df_energy_and_time.loc[(df_energy_and_time['algorithm'] == algorithm) & 
                                                    (df_energy_and_time['matrix_size'] == matrix) & 
                                                    (df_energy_and_time['tile_size'] == tile) & 
                                                    (df_energy_and_time['case'] == case), 
                                                    column_name] = df_counters_reduced.loc[(df_counters_reduced['algorithm'] == algorithm) & 
                                                                                            (df_counters_reduced['matrix_size'] == matrix) & 
                                                                                            (df_counters_reduced['tile_size'] == tile) &
                                                                                            (df_counters_reduced['task'] == task),
                                                                                            metrics[i]].values[0]
                        for task in original_frequency_tasks:
                            column_name = task_dict[task] + "_" + metrics[i]
                              
                            df_energy_and_time.loc[(df_energy_and_time['algorithm'] == algorithm) & 
                                                    (df_energy_and_time['matrix_size'] == matrix) & 
                                                    (df_energy_and_time['tile_size'] == tile) & 
                                                    (df_energy_and_time['case'] == case), 
                                                    column_name] = df_counters_original.loc[(df_counters_original['algorithm'] == algorithm) & 
                                                                                            (df_counters_original['matrix_size'] == matrix) & 
                                                                                            (df_counters_original['tile_size'] == tile) &
                                                                                            (df_counters_original['task'] == task),
                                                                                            metrics[i]].values[0]
    return df_energy_and_time

if __name__ == '__main__':
    df_energy_and_time, df_counters_original, df_counters_reduced = read_file()
    
    tasks, task_dict = extract_tasks(df_counters_original)
    
    output_df = logic(df_energy_and_time, df_counters_original, df_counters_reduced, task_dict)
    
    output_df.to_csv('final_cholesky_32768.csv', index=False)
    