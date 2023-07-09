import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as ticker
import glob
import os

def plot_graphs(folder, algo, matrix):
    TOLERANCE = float(os.getenv('TOLERANCE'))
    if TOLERANCE == None:
        print("TOLERANCE not defined")
        exit(0)

    file_format = f"energy_data_{algo}_{matrix}_*.csv"
    all_files = glob.glob(folder + file_format)

    tmp_df = []
    for file in all_files:
        df = pd.read_csv(file)
        tmp_df.append(df)

    merge_df = pd.concat(tmp_df)
    data = (merge_df.groupby(["algorithm", "matrix_size", "tile_size", "case", "threads", "task"]).mean().reset_index())

    plt.style.use("ggplot")

    fig, ax = plt.subplots(3, 4, figsize=(25, 15))

    for i, tile_size in enumerate(data["tile_size"].unique()):
        matrix_size = data["matrix_size"].unique()
        tile_data = data[data["tile_size"] == tile_size]

        grouped_data = tile_data.groupby("case").first()

        grouped_data["sum"] = grouped_data[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(
            axis=1
        )/1e6
        edp_default = grouped_data.loc[1, "sum"]
        mask1 = (grouped_data["sum"] == edp_default)

        ax[0, i].bar(
            grouped_data[mask1].index,
            grouped_data[mask1]["sum"],
            label="Baseline",
            color="#377eb8",
        )
        ax[0, i].bar(
            grouped_data[~mask1].index,
            grouped_data[~mask1]["sum"],
            label="Consumed energy",
            color="#4daf4a",
        )
        ax[0, i].axhline(y=edp_default, color="#377eb8", linestyle="--")
        ax[0, i].set_ylabel("Energy (J)")
        ax[0, i].legend(loc='lower right')
        ax[0, i].set_title(f"Matrix {matrix_size[0]} Tile {tile_size}")
        ax[0, i].grid(True)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        ax[0, i].yaxis.set_major_formatter(formatter)

        ######

        edp_default = grouped_data.loc[1, "time"]
        edp_default2 = grouped_data.loc[1, "time"] * TOLERANCE

        mask1 = (grouped_data["time"] == grouped_data.loc[1, "time"])

        ax[1, i].bar(
            grouped_data[mask1].index,
            grouped_data[mask1]["time"],
            label="Baseline",
            color="#377eb8",
        )
        ax[1, i].bar(
            grouped_data[~mask1].index,
            grouped_data[~mask1]["time"],
            label="Execution time",
            color="#984ea3",
        )

        ax[1, i].axhline(y=edp_default, color="#377eb8", linestyle="--")
        ax[1, i].axhline(y=edp_default2, color="#ff7f00", linestyle="--")

        ax[1, i].set_ylabel("Time (s)")
        ax[1, i].legend(loc='lower right')
        ax[1, i].grid(True)

        ######

        grouped_data["edp"] = grouped_data["sum"] * grouped_data["time"]
        edp_default = grouped_data.loc[1, "edp"]

        energy_default = grouped_data.loc[1, "sum"]
        time_default = grouped_data.loc[1, "time"] * TOLERANCE

        mask1 = (grouped_data["sum"] == energy_default) & (grouped_data["time"] == grouped_data.loc[1, "time"])
        mask2 = ~mask1 & ((grouped_data["sum"] > energy_default) | (grouped_data["time"] > time_default))
        mask3 = ~(mask1 | mask2)

        ax[2, i].bar(
            grouped_data[mask2].index, grouped_data[mask2]["edp"], label="Non acceptable EDP", color="#a65628"
        )
        ax[2, i].bar(
            grouped_data[mask3].index, grouped_data[mask3]["edp"], label="Acceptable EDP", color="#ff7f00"
        )
        ax[2, i].bar(
            grouped_data[mask1].index, grouped_data[mask1]["edp"], color="#377eb8", label="Baseline"
        )

        ax[2, i].axhline(y=edp_default, color="#377eb8", linestyle="--")
        ax[2, i].set_ylabel("J.s")
        ax[2, i].set_xlabel("Case")
        ax[2, i].legend(loc='lower right')
        ax[2, i].grid(True)
        ax[2, i].yaxis.set_major_formatter(formatter)

        ######

        case_labels = ["default" if case == 1 else f"c{case}" for case in grouped_data.index]
        ax[0, i].set_xticklabels([])
        ax[0, i].set_xticks([])
        ax[1, i].set_xticklabels([])
        ax[1, i].set_xticks([])
        ax[2, i].set_xticks(grouped_data.index)
        ax[2, i].set_xticklabels(case_labels, rotation=90)

    plt.subplots_adjust(hspace=0.1)
    plt.savefig(f"misc/results/graphs/raw/raw_all_{algo}_{matrix}_{TOLERANCE}.png")
    # plt.show()

def plot_graph_single(folder, algo, matrix, tile_size):

    TOLERANCE = float(os.getenv('TOLERANCE'))
    if TOLERANCE == None:
        print("TOLERANCE not defined")
        exit(0)

    file_format = f"energy_data_{algo}_{matrix}_*.csv"
    all_files = glob.glob(folder + file_format)

    tmp_df = []
    for file in all_files:
        df = pd.read_csv(file)
        tmp_df.append(df)

    merge_df = pd.concat(tmp_df)
    data = (merge_df.groupby(["algorithm", "matrix_size", "tile_size", "case", "threads", "task"]).mean().reset_index())

    data["sum"] = data[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1)/1e9

    plt.style.use("ggplot")
    plt.rcParams['font.size'] = 18

    fig, ax = plt.subplots(1, 3, figsize=(15, 6))

    matrix_size = data["matrix_size"].unique()
    tile_data = data[data["tile_size"] == tile_size]

    grouped_data = tile_data.groupby("case").first()

    edp_default = grouped_data.loc[1, "sum"]

    mask1 = (grouped_data["sum"] == edp_default)

    ax[0].bar(grouped_data[mask1].index, grouped_data[mask1]["sum"], color="#377eb8")
    ax[0].bar(
        grouped_data[~mask1].index,
        grouped_data[~mask1]["sum"],
        #label="Consumed energy",
        color="#4daf4a",
    )
    ax[0].axhline(y=edp_default, color="#377eb8", linestyle="dashed")
    ax[0].set_ylabel("Energy (kJ)")
    ax[0].legend(loc='lower right')
    ax[0].grid(True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    ax[0].yaxis.set_major_formatter(formatter)

    #########

    edp_default = grouped_data.loc[1, "time"]
    edp_default2 = grouped_data.loc[1, "time"] * TOLERANCE

    mask1 = (grouped_data["time"] == grouped_data.loc[1, "time"])

    ax[1].bar(grouped_data[mask1].index, grouped_data[mask1]["time"], color="#377eb8")
    ax[1].bar(
        grouped_data[~mask1].index,
        grouped_data[~mask1]["time"],
        # label="Execution time",
        color="#984ea3",
    )
    ax[1].axhline(y=edp_default, color="#377eb8", linestyle="dashed")
    ax[1].axhline(y=edp_default2, color="#ff7f00", linestyle="dashed", label='Time baseline + 10%')
    ax[1].set_ylabel("Time (s)")
    ax[1].legend(loc='lower right')
    ax[1].grid(True)
    ax[1].yaxis.set_major_formatter(formatter)

    #########

    grouped_data["edp"] = grouped_data["sum"] * grouped_data["time"]
    edp_default = grouped_data.loc[1, "edp"]

    energy_default = grouped_data.loc[1, "sum"]
    time_default = grouped_data.loc[1, "time"] * TOLERANCE

    grouped_data["edp"] = grouped_data["sum"] * grouped_data["time"]

    # Create boolean mask for your condition
    mask1 = (grouped_data["sum"] == energy_default) & (grouped_data["time"] == grouped_data.loc[1, "time"])
    mask2 = ~mask1 & ((grouped_data["sum"] > energy_default) | (grouped_data["time"] > time_default))
    mask3 = ~(mask1 | mask2)

    ax[2].bar(grouped_data[mask1].index, grouped_data[mask1]["edp"], color="#377eb8")
    ax[2].bar(grouped_data[mask2].index, grouped_data[mask2]["edp"], label="Non acceptable EDP", color="#a65628")
    ax[2].bar(grouped_data[mask3].index, grouped_data[mask3]["edp"], label="Acceptable EDP", color="#ff7f00")
    ##a65628

    ax[2].axhline(y=edp_default, color="#377eb8", linestyle="dashed")
    ax[2].set_ylabel("Energy x time")
    ax[2].set_xlabel("Case")
    ax[2].legend(loc='lower right')
    ax[2].grid(True)
    ax[2].yaxis.set_major_formatter(formatter)

    ax[1].set_title(f"Matrix size = {matrix_size[0]} x {matrix_size[0]} - Tile size = {tile_size} x {tile_size}")

    case_labels = ["default" if case == 1 else f"c{case}" for case in grouped_data.index]

    ax[0].set_xticks(grouped_data.index)
    ax[0].set_xticklabels(case_labels, rotation=90)
    ax[1].set_xticks(grouped_data.index)
    ax[1].set_xticklabels(case_labels, rotation=90)
    ax[2].set_xticks(grouped_data.index)
    ax[2].set_xticklabels(case_labels, rotation=90)

    plt.subplots_adjust(hspace=0, wspace=0.235)
    plt.subplots_adjust(bottom=0.17, top=0.93, left=0.06, right=0.99) 
    plt.savefig(f"misc/results/graphs/raw/raw_single_{algo}_{matrix}_{TOLERANCE}.png")
    # plt.show()


if __name__ == "__main__":
    key = int(sys.argv[1])

    folder = sys.argv[2]
    algo = sys.argv[3]
    matrix = sys.argv[4]

    if key == 1:
        plot_graphs(folder, algo, matrix)
    elif key == 2:
        tile = int(sys.argv[5])
        plot_graph_single(folder, algo, matrix, tile)
