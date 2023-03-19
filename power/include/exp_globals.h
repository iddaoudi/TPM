/*
 * =====================================================================================
 *
 *       Filename:  exp_globals.h
 *
 *    Description:  Global variables used across the project
 *
 *        Version:  1.0
 *        Created:  19/03/2023
 *       Revision:  none
 *       Compiler:  clang
 *
 *         Author:  Idriss Daoudi <idaoudi@anl.gov>
 *   Organization:  Argonne National Laboratory
 *
 * =====================================================================================
 */

#ifndef EXP_GLOBALS_H
#define EXP_GLOBALS_H

#define TPM_STRING_SIZE 16
#define TPM_TASK_STRING_SIZE 8
#define TPM_FILENAME_STRING_SIZE 16

extern int MAX_CPUS;
extern char *input_governor;
extern char *target_frequency;
extern int selected_case;
extern char *algorithm;

#endif // EXP_GLOBALS_H