/*
 * =====================================================================================
 *
 *       Filename:  sparselu.h
 *
 *    Description:  Control loop for SparseLU tasks power management
 *
 *        Version:  1.0
 *        Created:  03/01/2023
 *       Revision:  19/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void sparselu_control(int selected_case, char *task, int cpu,
                      unsigned long selected_frequency,
                      unsigned long original_frequency)
{
  static const char *task_names[] = {"lu0", "bdiv", "bmod", "fwd"};

  if (selected_case >= 1 && selected_case <= 15)
  {
    int task_mask = selected_case - 1;
    int task_found = 0;

    for (int i = 0; i < sizeof(task_names) / sizeof(task_names[0]); ++i)
    {
      if (!strcmp(task, task_names[i]) && (task_mask & (1 << i)))
      {
        tpm_set_max_frequency(cpu, selected_frequency);
        task_found = 1;
        break;
      }
    }

    if (!task_found)
    {
      tpm_set_max_frequency(cpu, original_frequency);
    }
  }
  else if (selected_case == 16)
  {
    tpm_set_max_frequency(cpu, original_frequency);
  }

  /* Set back the original frequency on the CPU used by a task */
  tpm_set_max_frequency(cpu, original_frequency);
}
