import pandas as pd
import data_treatment.dictionaries as dict


def logic(df_energy, df_counters):
    # A dataframe for each frequency, only 2 for now
    df_counters_high_freq = df_counters[
        df_counters["frequency"] == df_counters["frequency"].max()
    ]
    df_counters_low_freq = df_counters[
        df_counters["frequency"] == df_counters["frequency"].min()
    ]
    df_counters_high_freq = df_counters_high_freq.reset_index(drop=True)
    df_counters_low_freq = df_counters_low_freq.reset_index(drop=True)

    unique_algorithms = df_energy["algorithm"].unique()

    df_result = pd.DataFrame()

    for algorithm in unique_algorithms:
        # Cases to tasks dictionary
        algorithm_dict = dict.algorithm_cases_dict(algorithm)
        # All the algorithm tasks from the last entry of the dictionary
        algorithm_all_tasks = list(algorithm_dict.values())[-1]
        for case, tasks in algorithm_dict.items():
            # Create an energy dataframe for this case
            df_energy_case = df_energy[
                (df_energy["case"] == case) & (df_energy["algorithm"] == algorithm)
            ]
            df_energy_case = df_energy_case.reset_index(drop=True)
            for task in algorithm_all_tasks:
                # If task is in the dictionary: lower frequency
                if task in tasks:
                    df_to_merge = df_counters_low_freq[
                        (df_counters_low_freq["task"] == task)
                        & (df_counters_low_freq["algorithm"] == algorithm)
                    ]
                # If not, higher frequency
                else:
                    df_to_merge = df_counters_high_freq[
                        (df_counters_high_freq["task"] == task)
                        & (df_counters_high_freq["algorithm"] == algorithm)
                    ]
                # Merge the energy case dataframe with the selected metrics dataframe
                df_tmp = df_energy_case.merge(
                    df_to_merge, on=["algorithm", "matrix_size", "tile_size", "task"]
                )
                if df_tmp.empty:
                    print(
                        f"Merge result empty for algorithm={algorithm}, case={case}, task={task}"
                    )
                # Iterate
                df_result = pd.concat([df_result, df_tmp])

    # Define the columns that represent your metrics
    metric_columns = [
        "frequency",
        "PAPI_VEC_DP",
        "PAPI_L2_TCR",
        "PAPI_L3_TCR",
        "PAPI_TOT_CYC",
        "PAPI_RES_STL",
        "PAPI_L2_TCW",
        "PAPI_L3_TCW",
        "PAPI_TOT_INS",
        "PAPI_L3_TCM",
        "ilp",
        "cpi",
        "cmr",
        "vr",
        "scr",
    ]

    # Multiply each metric by its weight
    weighted_metrics = df_result.apply(
        lambda row: row[metric_columns] * row["weight"], axis=1
    )

    # Add the non-metric columns to our new DataFrame
    for col in [
        "algorithm",
        "matrix_size",
        "tile_size",
        "case",
        "PKG1",
        "PKG2",
        "DRAM1",
        "DRAM2",
        "time",
        "l3_cache_size",
        "number_of_tasks",
    ]:
        weighted_metrics[col] = df_result[col]

    # Compute the sum of the weighted metrics for each case
    case_metrics = (
        weighted_metrics.groupby(
            [
                "algorithm",
                "matrix_size",
                "tile_size",
                "case",
                "PKG1",
                "PKG2",
                "DRAM1",
                "DRAM2",
                "time",
                "l3_cache_size",
                "number_of_tasks",
            ]
        )
        .sum()
        .reset_index()
    )

    return case_metrics
