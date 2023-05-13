import pandas as pd
import data_treatment.utils as utils
import data_treatment.merge as merge
import learning.single_target_model as stm
import learning.two_targets_model as ttm


if __name__ == "__main__":
    # Compute the mean of the input files
    df_energy, df_counters = utils.mean_of_files()

    # Tasks abstraction
    tasks, task_dict = utils.extract_tasks(df_counters)
    print(task_dict)

    # Get the weights ratio
    df_counters = utils.calculate_task_ratio(df_counters)

    # Make the correspondance between cases and counters (depending on the frequency) and multiply by the weight
    final_df = merge.logic(df_energy, df_counters, task_dict)

    # Try models
    stm.single_target_model(final_df)
    ttm.two_targets_model(final_df)
