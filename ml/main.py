import pandas as pd
import data_treatment.utils as utils
import data_treatment.merge as merge
import plot.plot as plotter
import learning.single_target_classifier as stm
import learning.single_target_edp as edp
import learning.two_targets_classifier as ttm
import sys


if __name__ == "__main__":
    command = int(sys.argv[1])
    algo = sys.argv[2]

    array_of_df = []
    architecture = ""

    for i in range(3, len(sys.argv)):
        folder = sys.argv[i]
        tmp = folder.split("/")
        architecture = tmp[1]

        # Compute the mean of the input files
        df_energy, df_counters = utils.mean_of_files(folder)

        # Get the weights ratio
        df_counters = utils.calculate_task_ratio(df_counters)

        # Compute the new metrics generated by the counters
        df_counters = utils.calculate_new_columns(df_counters)

        # Make the correspondance between cases and counters (depending on the frequency) and multiply by the weight
        final_df = merge.logic(df_energy, df_counters)

        final_df = utils.normalize(final_df)

        # Perform Shapiro-Wilk test
        # utils.shapiro_wilk_test(final_df)
        # exit(0)

        # Append dataframe to array of dataframes
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
        if command == 0:
            plotter.plot_multi(combined_df, architecture)
            # plotter.plot_function(combined_df, architecture)
    else:
        print("Empty dataframe!")

    # Try models
    if command == 1:
        train_algorithms = [algo]
        train_matrix_sizes = [8192, 10240, 12288, 14336]
        test_algorithms = [algo]
        test_matrix_sizes = [16384, 18432, 20480, 22528, 24576]

        stm_var = 0
        edp_var = 1
        ttm_var = 0

        if stm_var == 1:
            stm.single_target_model(
                combined_df,
                train_algorithms,
                train_matrix_sizes,
                test_algorithms,
                test_matrix_sizes,
            )

        if edp_var == 1:
            edp.single_target_model_regression(
                combined_df,
                train_algorithms,
                train_matrix_sizes,
                test_algorithms,
                test_matrix_sizes,
                architecture,
            )

        if ttm_var == 1:
            ttm.two_targets_model(final_df)
