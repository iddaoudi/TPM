import pandas as pd
import matplotlib.pyplot as plt


def compute_gflops(df):
    tile_squared = df["tile"] ** 2
    df.loc[df["task"] == "gemm", "flops"] = 2 * tile_squared**3
    df.loc[df["task"] == "syrk", "flops"] = tile_squared**3
    df.loc[df["task"] == "trsm", "flops"] = tile_squared**3
    df.loc[df["task"] == "potrf", "flops"] = (1 / 3) * tile_squared**3

    # Compute the weights for each task
    total_weights = df.groupby("tile")["weight"].sum()
    df["weight_normalized"] = df.apply(
        lambda row: row["weight"] / total_weights[row["tile"]], axis=1
    )

    df["gflops"] = (df["flops"] * df["weight"]) / (df["time"]) * 1e-9

    # Compute the weighted GFLOPS for each task
    df["gflops_weighted"] = df["gflops"]  # * df["weight_normalized"]

    total_time_mean = df.groupby("tile")["total"].mean()

    total_df_list = []
    unique_tiles = df["tile"].unique()
    unique_matrix = df["matrix"].unique()
    for tile in unique_tiles:
        total_flops = (4 + 1 / 3) * unique_matrix**6
        total_gflops = total_flops / total_time_mean[tile] * 1e-9
        total_df_list.append({"tile": tile, "task": "total", "gflops": total_gflops})

    total_df = pd.DataFrame(total_df_list)

    df = pd.concat([df, total_df], ignore_index=True)

    # Compute the weighted GFLOPS for each tile size
    weighted_gflops = df.groupby("tile")["gflops_weighted"].sum().reset_index()
    weighted_gflops["task"] = "weighted"
    df = pd.concat([df, weighted_gflops], ignore_index=True)

    return df


def plot(gflops_df):
    # Create new dataframes for 'weighted' and 'total'
    total_df = gflops_df[gflops_df["task"] == "total"][["tile", "gflops"]].copy()
    total_df["task"] = "total"

    weighted_df = gflops_df.groupby("tile")["gflops_weighted"].sum().reset_index()
    weighted_df["task"] = "weighted"
    weighted_df["gflops"] = weighted_df["gflops_weighted"]
    weighted_df = weighted_df[["tile", "gflops", "task"]]

    # Concatenate the new dataframes with the original one, excluding 'total' and 'weighted' rows
    gflops_df = pd.concat(
        [gflops_df[gflops_df["task"] != "total"], total_df, weighted_df],
        ignore_index=True,
    )

    # Plot the data
    fig, ax = plt.subplots()
    for task in ["gemm", "trsm", "syrk", "potrf", "total", "weighted"]:
        data = gflops_df[gflops_df["task"] == task]
        ax.plot(data["tile"], data["gflops"], label=task)

    ax.set_xlabel("Tile Size")
    ax.set_ylabel("GFLOPs")
    ax.legend(loc="upper left")
    ax.set_title("Performance vs Tile Size for Different Tasks")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("potrf_times.csv")
    gflops_df = compute_gflops(df)
    plot(gflops_df)
