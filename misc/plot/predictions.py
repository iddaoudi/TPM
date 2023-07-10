import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.legend import Legend
import os
import sys

def plot_best_predictions(predictions, df, architecture):
    models = predictions["model"].unique()
    algorithms = predictions["algorithm"].unique()
    matrix_sizes = predictions["matrix_size"].unique()
    tile_sizes = predictions["tile_size"].unique()
    width = 0.7 / len(tile_sizes)  # Adjust width of the bars

    sns.set_theme()
    sns.set(font_scale=2)

    fig, ax = plt.subplots(figsize=(15, 10))

    # Generate color palette
    palette = sns.color_palette("hls", len(models))

    color_dict = dict(zip(models, palette))

    # Predefined hatch patterns
    hatch_patterns = ["///", "\\\\\\", "|||", "-", "+", "x", "o", "O", ".", "*"]
    tile_hatch_dict = dict(zip(tile_sizes, hatch_patterns))

    legend_patches = {}

    TOLERANCE = float(os.getenv('TOLERANCE'))
    if TOLERANCE == None:
        print("TOLERANCE not defined")
        exit(0)

    TARGET = os.getenv('TARGET')
    if TARGET == None:
        sys.exit("TARGET not defined")

    PLOT = os.getenv('PLOT')
    if PLOT == None:
        sys.exit("PLOT not defined")

    for algorithm in algorithms:
        for idx, matrix_size in enumerate(matrix_sizes):
            offset = (len(tile_sizes) - 1) / 2 * width

            for tile_pos, tile_size in enumerate(tile_sizes):
                best_improvement = None
                best_model = None
                default_case_edp = None
                default_case_energy = None
                default_case_time = None

                for model in models:
                    model_predictions = predictions[
                        (predictions["model"] == model)
                        & (predictions["algorithm"] == algorithm)
                        & (predictions["matrix_size"] == matrix_size)
                        & (predictions["tile_size"] == tile_size)
                    ]

                    default_case = df[
                        (df["case"] == 1)
                        & (df["algorithm"] == algorithm)
                        & (df["matrix_size"] == matrix_size)
                        & (df["tile_size"] == tile_size)
                    ]
                    # default_case_edp = default_case["edp"].values[0]
                    # default_case_energy = default_case["energy"].values[0]
                    # default_case_time = default_case["time"].values[0]

                    default_case_predicted_value = default_case["predicted_value"].values[0]

                    # case_edp = model_predictions["edp"]
                    # case_energy = model_predictions["energy"]
                    
                    predicted_value = model_predictions["predicted_value"]

                    if not predicted_value.empty:
                        current_improvement = (
                            (default_case_predicted_value - predicted_value.iloc[0])
                            / default_case_predicted_value
                        ) * 100

                        if (
                            best_improvement is None
                            or current_improvement > best_improvement
                        ):
                            best_improvement = current_improvement
                            best_model = model

                    # if TARGET == "edp":
                    #     if not case_edp.empty:
                    #         current_improvement = (
                    #             (default_case_edp - case_edp.iloc[0])
                    #             / default_case_edp
                    #         ) * 100

                    #         if (
                    #             best_improvement is None
                    #             or current_improvement > best_improvement
                    #         ):
                    #             best_improvement = current_improvement
                    #             best_model = model
                    # elif TARGET == "energy":
                    #     if not case_energy.empty:
                    #         current_improvement = (
                    #             (default_case_energy - case_energy.iloc[0])
                    #             / default_case_energy
                    #         ) * 100

                    #         if (
                    #             best_improvement is None
                    #             or current_improvement > best_improvement
                    #         ):
                    #             best_improvement = current_improvement
                    #             best_model = model
                    # else:
                    #     sys.exit("TARGET option unknown")

                if best_improvement is not None:
                    best_model_df = predictions[
                        (predictions["model"] == best_model)
                        & (predictions["algorithm"] == algorithm)
                        & (predictions["matrix_size"] == matrix_size)
                        & (predictions["tile_size"] == tile_size)
                    ]
                    best_model_energy = best_model_df["energy"].iloc[0]
                    best_model_time = best_model_df["time"].iloc[0]

                    if PLOT == "energy":
                        best_improvement = (
                                    (default_case_energy - best_model_energy)
                                    / default_case_energy
                                ) * 100
                    elif PLOT == "time":
                        best_improvement = (
                                    (default_case_time - best_model_time)
                                    / default_case_time
                                ) * 100
                    else:
                        sys.exit("PLOT option unknown")

                    bar_pos = idx - offset + tile_pos * width  # calculate bar position
                    bar = ax.bar(
                        bar_pos,
                        best_improvement,
                        width=width,
                        color=color_dict[best_model],
                        hatch=tile_hatch_dict[tile_size],
                    )  # Add hatch pattern based on tile size
                    legend_patches[best_model] = Patch(
                        color=color_dict[best_model]
                    )  # Create patch for legend

    ax.set_title(f"Best {PLOT} improvement for each matrix and tile sizes", pad=20)
    ax.set_ylabel("Improvement % over Default Case", labelpad=15)
    ax.set_xlabel("Matrix Size", labelpad=15)
    ax.set_xticks(np.arange(len(matrix_sizes)))
    ax.set_xticklabels(matrix_sizes)

    legend1 = ax.legend(
        legend_patches.values(),
        legend_patches.keys(),
        title="Models",
        # title_fontsize="13",
        loc="upper left",
        bbox_to_anchor=(1, 1)
    )
    ax.add_artist(legend1)

    legend_patches_tile = [
        Patch(facecolor="gray", hatch=tile_hatch_dict[size], edgecolor="black")
        for size in tile_sizes
    ]
    legend2 = Legend(
        ax,
        legend_patches_tile,
        labels=tile_sizes,
        title="Tile Sizes",
        loc="upper right",
        frameon=True,
        bbox_to_anchor=(1.2, 0.5)
    )
    ax.add_artist(legend2)

    ax.grid(True, linestyle="--", alpha=0.6)

    if PLOT == "time":
        ax.axhline(y=-5, color='r', linestyle='--')

    fig.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(f"misc/results/graphs/predictions/{TARGET}/predictions_{architecture}_{algorithm}_{TOLERANCE}_{PLOT}.png")
    # plt.show()

if __name__ == "__main__":
    file_data = sys.argv[1]
    file_pred = sys.argv[2]
    tmp = file_data.split("_")
    architecture = tmp[1]

    df = pd.read_csv(file_data)
    predictions = pd.read_csv(file_pred)

    plot_best_predictions(predictions, df, architecture)
