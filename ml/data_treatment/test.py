import pandas as pd

# Your DataFrames
df1 = pd.DataFrame(
    {
        "algorithm": ["a", "a", "a", "a", "a", "a", "a", "a"],
        "matrix": [10, 10, 10, 10, 10, 10, 10, 10],
        "case": [1, 1, 2, 2, 3, 3, 4, 4],
        "task": ["gemm", "trsm", "gemm", "trsm", "gemm", "trsm", "gemm", "trsm"],
    }
)

df2 = pd.DataFrame(
    {
        "algorithm": ["a", "a"],
        "matrix": [10, 10],
        "task": ["gemm", "trsm"],
        "value1": [11, 33],
        "value2": [22, 44],
    }
)

df3 = pd.DataFrame(
    {
        "algorithm": ["a", "a"],
        "matrix": [10, 10],
        "task": ["gemm", "trsm"],
        "value1": [12, 34],
        "value2": [23, 45],
    }
)

# Your Dictionary
dict_cases = {
    1: [],
    2: ["gemm"],
    3: ["trsm"],
    4: ["gemm", "trsm"],
}

# An empty DataFrame to store the results
df_result = pd.DataFrame()

for case, tasks in dict_cases.items():
    df1_case = df1[df1["case"] == case]

    for task in ["gemm", "trsm"]:
        if task in tasks:
            df_temp = df1_case[df1_case["task"] == task].merge(
                df3[df3["task"] == task], on=["algorithm", "matrix", "task"]
            )
        else:
            df_temp = df1_case[df1_case["task"] == task].merge(
                df2[df2["task"] == task], on=["algorithm", "matrix", "task"]
            )

        df_result = pd.concat([df_result, df_temp])

print(df_result)
