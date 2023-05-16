import pandas as pd
import data_treatment.utils as utils
import data_treatment.merge as merge
import plot.plot as plotter
import learning.single_target_model as stm
import learning.two_targets_model as ttm
import sys


if __name__ == "__main__":
    plot = 1
    models = 0

    array_of_df = []
    architecture = ""

    for i in range(1, len(sys.argv)):
        folder = sys.argv[i]
        tmp = folder.split("/")
        architecture = tmp[1]

        # Compute the mean of the input files
        df_energy, df_counters = utils.mean_of_files(folder)

        # Tasks abstraction
        tasks, task_dict = utils.extract_tasks(df_counters)
        print(task_dict)

        # Get the weights ratio
        df_counters = utils.calculate_task_ratio(df_counters)

        # Make the correspondance between cases and counters (depending on the frequency) and multiply by the weight
        final_df = merge.logic(df_energy, df_counters, task_dict)
        array_of_df.append(final_df)

    # Multiple algorithms
    if len(array_of_df) > 1:
        print("Merged dataframes!")
        combined_df = pd.concat(array_of_df, ignore_index=True)

    # Single algorithm
    elif len(array_of_df) == 1:
        print("Single dataframe!")
        combined_df = array_of_df[0]
        # Plot
        if plot == 1:
            # plotter.plot_multi(combined_df, architecture)
            plotter.plot_function(combined_df, architecture)

    else:
        print("Empty dataframe!")

    # Try models
    if models == 1:
        train_algorithms = ["cholesky"]
        train_matrix_sizes = [16384]
        test_algorithms = ["cholesky"]
        test_matrix_sizes = [24576, 32768]

        stm.single_target_model(
            combined_df,
            train_algorithms,
            train_matrix_sizes,
            test_algorithms,
            test_matrix_sizes,
        )
        # ttm.two_targets_model(final_df)
