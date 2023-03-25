/*
 * =====================================================================================
 *
 *       Filename:  cholesky.h
 *
 *    Description:  Control loop for Cholesky tasks power management
 *
 *        Version:  1.0
 *        Created:  25/12/2022
 *       Revision:  25/03/2023
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

void cholesky_control(int selected_case, const char *task, int cpu,
                      unsigned long selected_frequency,
                      unsigned long original_frequency)
{
  static const char *task_names[] = {"potrf", "gemm", "trsm", "syrk"};

  if (selected_case >= 1 && selected_case <= 15)
  {
    // Use bitmask since we have 1 <= cases <= 15, therefore can be reprensented in bits
    int task_mask = selected_case - 1;
    int task_found = 0;
    // If task is found, update its CPU frequency to the desired value
    for (int i = 0; i < sizeof(task_names) / sizeof(task_names[0]); ++i)
    {
      if (!strcmp(task, task_names[i]) && (task_mask & (1 << i)))
      {
        tpm_set_max_frequency(cpu, selected_frequency);
        task_found = 1;
        break;
      }
    }
    // If not found, set the CPU frequency to the original value
    if (!task_found)
    {
      tpm_set_max_frequency(cpu, original_frequency);
    }
  }
  // Case 16, where all the tasks get their frequency reduced
  else if (selected_case == 16)
  {
    tpm_set_max_frequency(cpu, selected_frequency);
  }
}
