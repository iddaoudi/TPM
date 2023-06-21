import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.legend import Legend

marker_symbols = {
    1: "o",
    2: "s",
    3: "^",
    4: "d",
    5: "*",
    6: "p",
    7: "x",
    8: "h",
    9: "+",
    10: "v",
    11: ">",
    12: "<",
    13: "H",
    14: "D",
    15: "P",
    16: "X",
}


def plot_function(df, architecture):
    groups = df.groupby(["algorithm", "matrix_size"])
    tile_sizes = df["tile_size"].unique()
    colors = np.linspace(0, 1, num=16)

    for name, group in groups:
        fig, axes = plt.subplots(1, len(tile_sizes), figsize=(12, 4))

        for i, tile_size in enumerate(tile_sizes):
            case1 = group.loc[(group["case"] == 1) & (group["tile_size"] == tile_size)]
            x_val = case1["time"].iloc[0] * 1.05
            y_val = case1[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(axis=1).iloc[0]

            ax = axes[i]
            tile_group = group.loc[group["tile_size"] == tile_size]

            handles = []
            for _, row in tile_group.iterrows():
                marker_color = plt.cm.jet(colors[row["case"] - 1])
                handle = ax.scatter(
                    row["time"],
                    row[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(),
                    color=marker_color,
                    marker=marker_symbols[row["case"]],
                    label="c" + str(row["case"]),
                    s=95,
                    alpha=0.9,
                )
                handles.append(handle)

            ax.axvline(
                x_val, color="r", linestyle="--", label=f"DET + 15% ({x_val:.2f})"
            )
            ax.axhline(y_val, color="r", linestyle="--", label=f"DEC ({y_val:.2f})")

            ax.set_title(f"Tile Size {tile_size}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Energy (uJ)")
            ax.legend(
                handles,
                [f"c{i}" for i in range(1, 17)],
                loc="center left",
                fancybox=True,
                shadow=True,
                bbox_to_anchor=(1, 0.5),
            )

        fig.suptitle(f"Algorithm: {name[0]} - Matrix size: {name[1]}")
        plt.subplots_adjust(top=0.85, bottom=0.1, wspace=0.4)

        fig.tight_layout()
        plt.savefig(
            f"./model/plot/figures/points_{architecture}_{name[0]}_{name[1]}.png",
            bbox_inches="tight",
        )
        # plt.show()


def plot_multi(data, architecture):
    matrix_sizes = data["matrix_size"].unique()
    tile_sizes = data["tile_size"].unique()
    algorithms = data["algorithm"].unique()

    data["energy"] = data["PKG1"] + data["PKG2"] + data["DRAM1"] + data["DRAM2"]
    data["time_energy_product"] = data["time"] * data["energy"]

    for algorithm in algorithms:
        for matrix_size in matrix_sizes:
            fig, axes = plt.subplots(
                1, len(tile_sizes), figsize=(5 * len(tile_sizes), 5), sharey=True
            )
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.suptitle(f"Algorithm: {algorithm} - Matrix Size: {matrix_size}")

            for index, tile_size in enumerate(tile_sizes):
                filtered_data = data[
                    (data["matrix_size"] == matrix_size)
                    & (data["tile_size"] == tile_size)
                ]

                case_1_time = filtered_data.loc[
                    filtered_data["case"] == 1, "time"
                ].values[0]
                case_1_energy = filtered_data.loc[
                    filtered_data["case"] == 1, "energy"
                ].values[0]

                condition = (filtered_data["energy"] < case_1_energy) & (
                    filtered_data["time"] <= case_1_time * 1.15
                )
                # If no case is fulfilling the 2 constraints
                if not condition.any():
                    continue
                best_case = (
                    filtered_data.loc[condition].sort_values(["energy", "time"]).iloc[0]
                )

                def color_map(row):
                    if row.name == best_case.name:
                        return "green"
                    elif (
                        row["energy"] < case_1_energy
                        and row["time"] <= case_1_time * 1.15
                    ):
                        return "red"
                    else:
                        return "black"

                colors = filtered_data.apply(color_map, axis=1)

                ax1 = axes[index]
                ax1.bar(
                    filtered_data["case"],
                    filtered_data["time_energy_product"],
                    color=colors,
                )

                ax1.set_ylabel("Time * Energy")
                ax1.set_title(f"Tile Size {tile_size}")

            fig.tight_layout()
            plt.savefig(f"{architecture}_{algorithm}_{matrix_size}.png")
            plt.show()


color_dict = {
    "LR": "blue",
    "Ridge": "green",
    "Lasso": "red",
    "GB": "purple",
    "XGBoost": "orange",
    "CatBoost": "magenta",
}


def plot_predictions(predictions, df, architecture):
    models = predictions["model"].unique()
    algorithms = predictions["algorithm"].unique()
    matrix_sizes = predictions["matrix_size"].unique()
    tile_sizes = predictions["tile_size"].unique()
    width = 0.1  # the width of the bars

    df["time_energy_product"] = df["time"] * df[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(
        axis=1
    )

    for algorithm in algorithms:
        for matrix_size in matrix_sizes:
            fig, ax = plt.subplots(figsize=(10, 7))  # single ax
            fig.suptitle(f"Algorithm: {algorithm} - Matrix Size: {matrix_size}")

            for idx, model in enumerate(models):  # iterate over models instead of tiles
                model_predictions = predictions[
                    (predictions["model"] == model)
                    & (predictions["algorithm"] == algorithm)
                    & (predictions["matrix_size"] == matrix_size)
                ]
                bar_positions = (
                    np.arange(len(tile_sizes)) + idx * width
                )  # new bar position for each model

                for tile_size_pos, tile_size in zip(bar_positions, tile_sizes):
                    tile_predictions = model_predictions[
                        model_predictions["tile_size"] == tile_size
                    ]

                    # Extract the default case
                    default_case = df[
                        (df["case"] == 1)
                        & (df["algorithm"] == algorithm)
                        & (df["matrix_size"] == matrix_size)
                        & (df["tile_size"] == tile_size)
                    ]
                    default_case_value = default_case["time_energy_product"].values[0]

                    # Calculate the improvement percentage over the default case
                    best_case_value = tile_predictions["time"] * tile_predictions[
                        ["PKG1", "PKG2", "DRAM1", "DRAM2"]
                    ].sum(axis=1)
                    if not best_case_value.empty:
                        improvement_percentage = (
                            (default_case_value - best_case_value) / default_case_value
                        ) * 100
                        bar = ax.bar(
                            tile_size_pos,  # plot position
                            improvement_percentage,
                            width=width,  # set bar width
                            color=color_dict[model],  # use color from color_dict
                        )

                # Add the legend for the model just once
                bar.set_label(model)

            ax.set_title(f"Tile Size Improvement")
            ax.set_ylabel("Improvement % over Default Case")
            ax.set_xticks(np.arange(len(tile_sizes)) + width * len(models) / 2)
            ax.set_xticklabels(tile_sizes)
            ax.legend()  # add this line
            fig.tight_layout()
            plt.savefig(
                f"predictions_{architecture}_{algorithm}_{matrix_size}_robust.png"
            )
            plt.show()


def plot_best_predictions(predictions, df, architecture):
    models = predictions["model"].unique()
    algorithms = predictions["algorithm"].unique()
    matrix_sizes = predictions["matrix_size"].unique()
    tile_sizes = predictions["tile_size"].unique()
    width = 0.35 / len(tile_sizes)  # Adjust width of the bars

    # df["time_energy_product"] = df["time"] * df[["PKG1", "PKG2", "DRAM1", "DRAM2"]].sum(
    #     axis=1
    # )

    # Use seaborn styles
    sns.set_theme()
    plt.figure(figsize=(10, 7))  # Adjust as needed
    fig, ax = plt.subplots()

    # Increase font size
    plt.rcParams.update({"font.size": 14})

    # Generate color palette
    palette = sns.color_palette("hls", len(models))

    color_dict = dict(zip(models, palette))

    # Predefined hatch patterns
    hatch_patterns = ["///", "\\\\\\", "|||", "+", "-", "x", "o", "O", ".", "*"]
    tile_hatch_dict = dict(zip(tile_sizes, hatch_patterns))

    legend_patches = {}  # Initialize dictionary for legend patches

    for algorithm in algorithms:
        for idx, matrix_size in enumerate(matrix_sizes):
            offset = (len(tile_sizes) - 1) / 2 * width

            for tile_pos, tile_size in enumerate(tile_sizes):
                best_improvement = None
                best_model = None

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
                    default_case_value = default_case["edp"].values[0]
                    # default_case_value = default_case["time_energy_product"].values[0]

                    # best_case_value = model_predictions["time"] * model_predictions[
                    #     ["PKG1", "PKG2", "DRAM1", "DRAM2"]
                    # ].sum(axis=1)
                    best_case_value = model_predictions["edp"]

                    if not best_case_value.empty:
                        current_improvement = (
                            (default_case_value - best_case_value.iloc[0])
                            / default_case_value
                        ) * 100

                        if (
                            best_improvement is None
                            or current_improvement > best_improvement
                        ):
                            best_improvement = current_improvement
                            best_model = model

                if best_improvement is not None:
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

    ax.set_title("Best improvement for each matrix and tile sizes", pad=20)
    ax.set_ylabel("Improvement % over Default Case", labelpad=15)
    ax.set_xlabel("Matrix Size", labelpad=15)
    ax.set_xticks(np.arange(len(matrix_sizes)))
    ax.set_xticklabels(matrix_sizes)

    legend1 = ax.legend(
        legend_patches.values(),
        legend_patches.keys(),
        title="Models",
        title_fontsize="13",
        loc="upper left",
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
    )
    ax.add_artist(legend2)

    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.savefig(f"predictions_{architecture}_{algorithm}_robust.png")
    plt.show()
