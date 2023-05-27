import matplotlib.pyplot as plt
import numpy as np

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

    metric = ""

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
                    filtered_data["time"] <= case_1_time * 1.05
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
                        and row["time"] <= case_1_time * 1.05
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
                # ax1.scatter(
                #     filtered_data["case"],
                #     filtered_data["time_energy_product"],
                #     c=colors,
                #     marker="o",
                # )
                # ax1.plot(
                #     filtered_data["case"],
                #     filtered_data["time_energy_product"],
                #     c="gray",
                #     linewidth=0.5,
                # )
                ax1.set_ylabel("Time * Energy")
                ax1.set_title(f"Tile Size {tile_size}")

                if len(metric) != 0:
                    ax2 = ax1.twinx()
                    ax2.scatter(
                        filtered_data["case"],
                        filtered_data[metric],
                        c=colors,
                        marker="x",
                    )
                    ax2.set_ylabel(metric)
                    ax2.yaxis.set_label_position("right")
                    ax2.yaxis.tick_right()

                    max_value = filtered_data[metric].max()
                    min_value = filtered_data[metric].min()

                    axes[index].set_title(f"Tile Size: {tile_size}")
                    axes[index].set_xlabel("Case")
                    axes[index].set_xticks(filtered_data["case"])

            fig.tight_layout()
            if len(metric) != 0:
                plt.savefig(
                    f"./model/plot/figures/bars/{architecture}_{algorithm}_{matrix_size}_{metric}.png"
                )
            else:
                plt.savefig(
                    f"./model/plot/figures/bars/{architecture}_{algorithm}_{matrix_size}.png"
                )
            # plt.show()
