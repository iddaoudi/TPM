import sys
import pandas as pd
import dictionaries

metrics = ['mem_boundness','arithm_intensity','ilp','l3_cache_ratio']

def calculate_task_ratio(): # FIXME Only Cholesky here
    # Number of tasks per algorithm file
    df = pd.read_csv(sys.argv[4])
    task_sum = df[['potrf', 'trsm', 'syrk', 'gemm']].sum(axis=1)
    ratios = df[['matrix_size', 'tile_size', 'potrf', 'trsm', 'syrk', 'gemm']].copy()
    ratios[['potrf', 'trsm', 'syrk', 'gemm']] = ratios[['potrf', 'trsm', 'syrk', 'gemm']].div(task_sum, axis=0)
    return ratios

def read_file():
    # mean file
    df_energy_and_time   = pd.read_csv(sys.argv[1])
    # counters file with original freq
    df_counters_original = pd.read_csv(sys.argv[2])
    # counters file with reduced freq
    df_counters_reduced  = pd.read_csv(sys.argv[3])
    return df_energy_and_time, df_counters_original, df_counters_reduced

def extract_tasks(df_counters):
    task_list = list({task for task in df_counters['task']})
    task_dict = {elem: f"task{idx+1}" for idx, elem in enumerate(task_list)}
    return task_list, task_dict

def logic(df_energy_and_time, df_counters_original, df_counters_reduced, task_dict, tasks_weights):
    unique_algorithm = df_energy_and_time['algorithm'].unique()
    unique_matrix    = df_energy_and_time['matrix_size'].unique()
    unique_tile      = df_energy_and_time['tile_size'].unique()
    unique_cases     = df_energy_and_time['case'].unique()
    
    # FIXME Only Cholesky here
    if unique_algorithm == 'cholesky':
        dico = dictionaries.cholesky_dict
    
    # Create the metrics columns with the abstract names
    for task in dico[16]:
        for metric in metrics:
            column_name = task_dict[task] + "_" + metric
            if column_name not in df_energy_and_time.columns:
                df_energy_and_time = df_energy_and_time.assign(**{column_name: 0})
    
    # Create the tasks columns to indicate which ones got their frequency reduced (for the model)
    for task in task_dict.values():
        df_energy_and_time[task] = 0

    for i in range(len(metrics)):
        for algorithm in unique_algorithm:
            for matrix in unique_matrix:
                for tile in unique_tile:
                    for case in unique_cases:
                        reduced_frequency_tasks = dico[case]
                        original_frequency_tasks = []
                        # First populate the list of task with original frequency ...
                        for t in dico[16]:
                            if t not in reduced_frequency_tasks:
                                original_frequency_tasks.append(t)
                        # ... then do the logic, reduced frequency tasks first
                        for task in reduced_frequency_tasks:
                            # Indicate which tasks got reduced frequency
                            df_energy_and_time.loc[(df_energy_and_time['algorithm'] == algorithm) & 
                                                    (df_energy_and_time['matrix_size'] == matrix) & 
                                                    (df_energy_and_time['tile_size'] == tile) & 
                                                    (df_energy_and_time['case'] == case), 
                                                    task_dict[task]] = 1
                            # Set the name of the column with the task name abstration and the metric name
                            column_name = task_dict[task] + "_" + metrics[i]
                            # Get the task weight
                            task_weight = tasks_weights.loc[(tasks_weights['matrix_size'] == matrix) & (tasks_weights['tile_size'] == tile), task].values[0]
                            # Find the metric and multiply it by the task weight
                            df_energy_and_time.loc[(df_energy_and_time['algorithm'] == algorithm) & 
                                                    (df_energy_and_time['matrix_size'] == matrix) & 
                                                    (df_energy_and_time['tile_size'] == tile) & 
                                                    (df_energy_and_time['case'] == case), 
                                                    column_name] = df_counters_reduced.loc[(df_counters_reduced['algorithm'] == algorithm) & 
                                                                                            (df_counters_reduced['matrix_size'] == matrix) & 
                                                                                            (df_counters_reduced['tile_size'] == tile) &
                                                                                            (df_counters_reduced['task'] == task),
                                                                                            metrics[i]].values[0] * task_weight
                        for task in original_frequency_tasks:
                            column_name = task_dict[task] + "_" + metrics[i]

                            task_weight = tasks_weights.loc[(tasks_weights['matrix_size'] == matrix) & (tasks_weights['tile_size'] == tile), task].values[0]
                              
                            df_energy_and_time.loc[(df_energy_and_time['algorithm'] == algorithm) & 
                                                    (df_energy_and_time['matrix_size'] == matrix) & 
                                                    (df_energy_and_time['tile_size'] == tile) & 
                                                    (df_energy_and_time['case'] == case), 
                                                    column_name] = df_counters_original.loc[(df_counters_original['algorithm'] == algorithm) & 
                                                                                            (df_counters_original['matrix_size'] == matrix) & 
                                                                                            (df_counters_original['tile_size'] == tile) &
                                                                                            (df_counters_original['task'] == task),
                                                                                            metrics[i]].values[0] * task_weight
    return df_energy_and_time
    

if __name__ == '__main__':
    # Get data
    df_energy_and_time, df_counters_original, df_counters_reduced = read_file()
    
    # Tasks abstraction
    tasks, task_dict = extract_tasks(df_counters_original)

    # Get the weight of every task
    tasks_weights = calculate_task_ratio()
        
    # Make the correspondance between the cases and the counters (depending on the freq) and multiply with the tasks weights
    output_df = logic(df_energy_and_time, df_counters_original, df_counters_reduced, task_dict, tasks_weights)
    
    output_df.to_csv('final_tmp_cholesky_16384.csv', index=False)
    