import pandas as pd
import data_treatment.dictionaries as dict


def logic(df_energy, df_counters, task_dict):
    unique_algorithms = df_energy["algorithm"].unique()
    unique_matrix = df_energy["matrix_size"].unique()
    unique_tiles = df_energy["tile_size"].unique()
    unique_cases = df_energy["case"].unique()

    # FIXME Cholesky and QR only
    for algorithm in unique_algorithms:
        if algorithm == "cholesky":
            cases_dict = dict.cholesky_dict
        elif algorithm == "qr":
            cases_dict = dict.qr_dict
        else:
            print("Missing dictionnary!")
            exit(1)

    # Create the metrics columns with the abstract names
    for task in cases_dict[len(cases_dict)]:
        for metric in dict.metrics:
            column_name = task_dict[task] + "_" + metric
            if column_name not in df_energy.columns:
                df_energy = df_energy.assign(**{column_name: 0})

    # Create the tasks columns that will indicate which ones got their frequency reduced
    for task in task_dict.values():
        df_energy[task] = 0

    for i in range(len(dict.metrics)):
        for algorithm in unique_algorithms:
            for matrix in unique_matrix:
                for tile in unique_tiles:
                    for case in unique_cases:
                        reduced_frequency_tasks = cases_dict[case]
                        original_frequency_tasks = []
                        # First populate the list of task with original frequency ...
                        for t in cases_dict[len(cases_dict)]:
                            if t not in reduced_frequency_tasks:
                                original_frequency_tasks.append(t)
                        # ... then do the logic, reduced frequency tasks first
                        for task in reduced_frequency_tasks:
                            # Get the frequency
                            frequency = df_counters.loc[
                                (df_counters["algorithm"] == algorithm)
                                & (df_counters["matrix_size"] == matrix)
                                & (df_counters["tile_size"] == tile)
                                & (df_counters["task"] == task),
                                "frequency",
                            ].values[0]
                            # Insert frequency
                            df_energy.loc[
                                (df_energy["algorithm"] == algorithm)
                                & (df_energy["matrix_size"] == matrix)
                                & (df_energy["tile_size"] == tile)
                                & (df_energy["case"] == case),
                                task_dict[task],
                            ] = (
                                frequency * 1e-6
                            )

                            # Set the name of the column with the task name abstration and the metric name
                            column_name = task_dict[task] + "_" + dict.metrics[i]

                            # Get the task weight
                            task_weight = df_counters.loc[
                                (df_counters["matrix_size"] == matrix)
                                & (df_counters["tile_size"] == tile)
                                & (df_counters["task"] == task),
                                "weight_ratio",
                            ].values[0]

                            # Find the metric according to the frequency and multiply it by the task weight
                            df_energy.loc[
                                (df_energy["algorithm"] == algorithm)
                                & (df_energy["matrix_size"] == matrix)
                                & (df_energy["tile_size"] == tile)
                                & (df_energy["case"] == case),
                                column_name,
                            ] = (
                                df_counters.loc[
                                    (df_counters["algorithm"] == algorithm)
                                    & (df_counters["matrix_size"] == matrix)
                                    & (df_counters["tile_size"] == tile)
                                    & (df_counters["task"] == task)
                                    & (df_counters["frequency"] == frequency),
                                    dict.metrics[i],
                                ].values[0]
                                * task_weight
                            )

                        for task in original_frequency_tasks:
                            # Get the frequency
                            frequency = df_counters.loc[
                                (df_counters["algorithm"] == algorithm)
                                & (df_counters["matrix_size"] == matrix)
                                & (df_counters["tile_size"] == tile)
                                & (df_counters["task"] == task),
                                "frequency",
                            ].values[1]
                            # Insert frequency
                            df_energy.loc[
                                (df_energy["algorithm"] == algorithm)
                                & (df_energy["matrix_size"] == matrix)
                                & (df_energy["tile_size"] == tile)
                                & (df_energy["case"] == case),
                                task_dict[task],
                            ] = (
                                frequency * 1e-6
                            )

                            # Set the name of the column with the task name abstration and the metric name
                            column_name = task_dict[task] + "_" + dict.metrics[i]

                            # Get the task weight
                            task_weight = df_counters.loc[
                                (df_counters["matrix_size"] == matrix)
                                & (df_counters["tile_size"] == tile)
                                & (df_counters["task"] == task),
                                "weight_ratio",
                            ].values[0]

                            # Find the metric according to the frequency and multiply it by the task weight
                            df_energy.loc[
                                (df_energy["algorithm"] == algorithm)
                                & (df_energy["matrix_size"] == matrix)
                                & (df_energy["tile_size"] == tile)
                                & (df_energy["case"] == case),
                                column_name,
                            ] = (
                                df_counters.loc[
                                    (df_counters["algorithm"] == algorithm)
                                    & (df_counters["matrix_size"] == matrix)
                                    & (df_counters["tile_size"] == tile)
                                    & (df_counters["task"] == task)
                                    & (df_counters["frequency"] == frequency),
                                    dict.metrics[i],
                                ].values[0]
                                * task_weight
                            )
    return df_energy
