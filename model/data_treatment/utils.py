import pandas as pd
import glob


def extract_tasks(df_counters):
    task_list = list({task for task in df_counters["task"]})
    task_dict = {elem: f"task{idx+1}" for idx, elem in enumerate(task_list)}
    return task_list, task_dict


def calculate_task_ratio(df_counters):
    grouped = df_counters.groupby(
        ["algorithm", "matrix_size", "tile_size", "frequency"]
    )
    weight_sum = grouped["weight"].transform("sum")
    df_counters["weight_ratio"] = df_counters["weight"] / weight_sum
    return df_counters


def mean_of_files(folder):
    energy_files_format = "energy_data_*.csv"
    counter_files_format = "counters_*.csv"

    energy_files_paths = glob.glob(folder + energy_files_format)
    counters_files_paths = glob.glob(folder + counter_files_format)

    energy_dataframes = []
    counters_dataframes = []

    for file in energy_files_paths:
        df = pd.read_csv(file)
        energy_dataframes.append(df)
        merge_dfs = pd.concat(energy_dataframes)
        mean_energy = (
            merge_dfs.groupby(["algorithm", "matrix_size", "tile_size", "case"])
            .mean()
            .reset_index()
        )

    for file in counters_files_paths:
        df = pd.read_csv(file)
        counters_dataframes.append(df)
        merge_dfs = pd.concat(counters_dataframes)
        mean_counters = (
            merge_dfs.groupby(
                ["algorithm", "matrix_size", "tile_size", "task", "frequency", "weight"]
            )
            .mean()
            .reset_index()
        )
    return mean_energy, mean_counters
