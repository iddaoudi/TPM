import pandas as pd
import glob
from scipy import stats
import numpy as np


def calculate_task_ratio(df_counters):
    grouped = df_counters.groupby(
        ["algorithm", "matrix_size", "tile_size", "frequency"]
    )
    weight_sum = grouped["weight"].transform("sum")
    df_counters["number_of_tasks"] = weight_sum
    df_counters["weight"] = df_counters["weight"] / weight_sum
    return df_counters


def calculate_new_columns(df_counters):
    df_counters["ilp"] = df_counters["PAPI_TOT_INS"] / df_counters["PAPI_TOT_CYC"]
    df_counters["cpi"] = df_counters["PAPI_TOT_CYC"] / df_counters["PAPI_TOT_INS"]
    df_counters["cmr"] = df_counters["PAPI_L3_TCM"] / df_counters["PAPI_L3_TCR"]
    df_counters["vr"] = df_counters["PAPI_VEC_DP"] / df_counters["PAPI_TOT_INS"]
    df_counters["scr"] = df_counters["PAPI_RES_STL"] / df_counters["PAPI_TOT_CYC"]

    df_counters["PAPI_VEC_DP"] = (
        df_counters["PAPI_VEC_DP"] / df_counters["PAPI_TOT_INS"]
    )
    df_counters["PAPI_L2_TCR"] = (
        df_counters["PAPI_L2_TCR"] / df_counters["PAPI_TOT_INS"]
    )
    df_counters["PAPI_L3_TCR"] = (
        df_counters["PAPI_L3_TCR"] / df_counters["PAPI_TOT_INS"]
    )
    df_counters["PAPI_RES_STL"] = (
        df_counters["PAPI_RES_STL"] / df_counters["PAPI_TOT_INS"]
    )
    df_counters["PAPI_L2_TCW"] = (
        df_counters["PAPI_L2_TCW"] / df_counters["PAPI_TOT_INS"].max()
    )
    df_counters["PAPI_L3_TCW"] = (
        df_counters["PAPI_L3_TCW"] / df_counters["PAPI_TOT_INS"].max()
    )
    df_counters["PAPI_L3_TCM"] = (
        df_counters["PAPI_L3_TCM"] / df_counters["PAPI_TOT_INS"]
    )

    return df_counters


def normalize(df):
    df["edp"] = df[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1) * df["time"]
    algorithms = df["algorithm"].unique()
    matrix_sizes = df["matrix_size"].unique()
    tile_sizes = df["tile_size"].unique()
    for algorithm in algorithms:
        for matrix in matrix_sizes:
            for tile in tile_sizes:
                default_case = df[
                    (df["case"] == 1)
                    & (df["algorithm"] == algorithm)
                    & (df["matrix_size"] == matrix)
                    & (df["tile_size"] == tile)
                ]
                if not default_case.empty:
                    default_edp = default_case["edp"].iloc[0]
                    df.loc[
                        (df["algorithm"] == algorithm)
                        & (df["matrix_size"] == matrix)
                        & (df["tile_size"] == tile),
                        "edp",
                    ] /= default_edp

    df["matrix_size_normalized"] = df["matrix_size"] / df["matrix_size"].max()
    df["tile_size_normalized"] = df["tile_size"] / df["tile_size"].max()
    df["number_of_tasks_normalized"] = (
        df["number_of_tasks"] / df["number_of_tasks"].max()
    )
    df["case_normalized"] = df["case"] / df["case"].max()
    df["frequency_normalized"] = df["frequency"] / df["frequency"].max()

    return df


def mean_of_files(folder):
    energy_files_format = "energy_data_*.csv"
    counter_files_format = "counters_*.csv"

    energy_files_paths = glob.glob(folder + energy_files_format)
    counters_files_paths = glob.glob(folder + counter_files_format)

    energy_dataframes = []
    counters_dataframes = []

    for file in energy_files_paths:
        df = pd.read_csv(file)
        df = df.dropna(how="all", axis=1)
        energy_dataframes.append(df)
    merge_dfs = pd.concat(energy_dataframes)
    mean_energy = (
        merge_dfs.groupby(
            ["algorithm", "matrix_size", "tile_size", "case", "threads", "task"]
        )
        .mean()
        .reset_index()
    )

    for file in counters_files_paths:
        df = pd.read_csv(file)
        df = df.dropna(how="all", axis=1)
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


def shapiro_wilk_test(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        data = df[column].sample(min(5000, len(df[column])))  # limit to 5000 samples
        _, p_value = stats.shapiro(data)

        if p_value > 0.05:
            print(f"Column {column} is likely normally distributed.")
        else:
            print(f"Column {column} is likely not normally distributed.")
