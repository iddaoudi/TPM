import pandas as pd
import matplotlib.pyplot as plt


def plot_task_time(df):
    df["time_per_weight"] = df["time"]  # / df["weight"]

    fig, ax = plt.subplots()
    for key, grp in df.groupby(["task"]):
        ax = grp.plot(
            ax=ax,
            kind="line",
            x="tile",
            y="time_per_weight",
            label=key[0] + str(" time * ") + key[0] + str(" weight"),
            marker=".",
        )

    df_total_time = df.drop_duplicates(subset="tile", keep="first")
    df_total_time.plot(
        ax=ax,
        kind="line",
        x="tile",
        y="total",
        label="Cholesky time",
        color="black",
        linewidth=2,
        marker="x",
        markersize=10,
    )

    # Compute the sum of 'task_time' for each 'tile'
    df_sum_task_time = df.groupby("tile")["time"].sum().reset_index()
    df_sum_task_time.rename(columns={"time": "sum_task_time"}, inplace=True)
    df_sum_task_time.plot(
        ax=ax,
        kind="line",
        x="tile",
        y="sum_task_time",
        label="Sum of task times * task weight",
        color="purple",
        linewidth=2,
        marker="o",
    )

    # Set the title and labels
    plt.xlabel("Tile Size")
    plt.ylabel("Task Time")
    plt.legend(loc="best")
    plt.grid()

    # Show the plot
    plt.savefig("task_times.png")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("potrf_times.csv")
    plot_task_time(df)
