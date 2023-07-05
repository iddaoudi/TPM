import pandas as pd
import matplotlib.pyplot as plt
import sys


def plot_graphs(csv_file):
    # read csv file into a pandas DataFrame
    data = pd.read_csv(csv_file)

    plt.style.use("ggplot")
    # create a figure and a 3x3 subplot layout
    fig, ax = plt.subplots(3, 4, figsize=(25, 15))

    # loop over the unique tile_sizes
    for i, tile_size in enumerate(data["tile_size"].unique()):
        # select data for the current tile_size
        matrix_size = data["matrix_size"].unique()
        tile_data = data[data["tile_size"] == tile_size]

        # group by 'case' and take the first record of each group as they are identical in terms of PKG1, PKG2, DRAM1, DRAM2 values
        grouped_data = tile_data.groupby("case").first()

        grouped_data["sum"] = grouped_data[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(
            axis=1
        )
        hor_value = grouped_data.loc[1, "sum"]
        ax[0, i].bar(
            grouped_data.index,
            grouped_data["sum"],
            label="Sum of PKGs and DRAMs",
            color="blue",
        )
        ax[0, i].axhline(y=hor_value, color="green", linestyle="--")
        ax[0, i].set_ylabel("Energy (uj)")
        ax[0, i].legend()
        ax[0, i].set_title(f"Matrix {matrix_size[0]} Tile {tile_size}")
        ax[0, i].grid(True)

        hor_value = grouped_data.loc[1, "time"]
        hor_value2 = grouped_data.loc[1, "time"] * 1.15
        ax[1, i].axhline(y=hor_value, color="green", linestyle="--")
        ax[1, i].axhline(y=hor_value2, color="red", linestyle="--")
        ax[1, i].bar(
            grouped_data.index,
            grouped_data["time"],
            label="Execution time",
            color="black",
        )
        ax[1, i].set_ylabel("Time (s)")
        ax[1, i].legend()
        ax[1, i].grid(True)

        grouped_data["edp"] = grouped_data["sum"] * grouped_data["time"]
        hor_value = grouped_data.loc[1, "edp"]
        hor_value2 = grouped_data.loc[1, "edp"] * 1.15
        ax[2, i].bar(
            grouped_data.index, grouped_data["edp"], label="EDP", color="orange"
        )
        ax[2, i].axhline(y=hor_value, color="green", linestyle="--")
        ax[2, i].axhline(y=hor_value2, color="red", linestyle="--")
        ax[2, i].set_ylabel("Energy * time")
        ax[2, i].set_xlabel("Case")
        ax[2, i].legend()
        ax[2, i].grid(True)

    plt.savefig("cholesky_cascade_24.png")
    plt.show()


if __name__ == "__main__":
    file = sys.argv[1]
    plot_graphs(file)
