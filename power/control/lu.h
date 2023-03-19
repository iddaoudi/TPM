/*
 * =====================================================================================
 *
 *       Filename:  lu.h
 *
 *    Description:  Control loop for LU tasks power management
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  19/03/2023
 *       Compiler:  gcc
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  idaoudi@anl.gov
 *
 * =====================================================================================
 */

void lu_control(int selected_case, const char *task, int cpu,
                unsigned long selected_frequency,
                unsigned long original_frequency)
{
  static const char *task_names[] = {"getrf", "gemm", "trsm"};

  if (selected_case >= 1 && selected_case <= 7)
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
  else if (selected_case == 8)
  {
    tpm_set_max_frequency(cpu, original_frequency);
  }

  /* Set back the original frequency on the CPU used by a task */
  tpm_set_max_frequency(cpu, original_frequency);
}
