import csv


def algorithm_cases_dict(algorithm):
    filename = "ml/data_treatment/algo_tasks.csv"
    with open(filename, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    all_data = {}
    current_algorithm = None
    for row in data:
        # Detect a new algorithm
        if not row:
            current_algorithm = None
            continue
        elif current_algorithm != row[0]:
            current_algorithm = row[0]
            all_data[current_algorithm] = {i: [] for i in range(1, int(row[1]) + 1)}

        # Fill dictionary for current algorithm
        count = int(row[1])
        all_data[current_algorithm][count] = row[2:]

    return all_data[algorithm]


metrics = [
    "PAPI_VEC_DP",
    "PAPI_L2_TCR",
    "PAPI_L3_TCR",
    # "PAPI_TOT_CYC",
    "PAPI_RES_STL",
    "PAPI_L2_TCW",
    "PAPI_L3_TCW",
    # "PAPI_TOT_INS",
    "PAPI_L3_TCM",
    "ilp",
    "cpi",
    "cmr",
    "vr",
    "scr",
]
