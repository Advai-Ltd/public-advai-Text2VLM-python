# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
Plotting utilities for relevancy scores.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def bootstrap_proportions(data, n_bootstrap=50, random_state=None):
    """Perform bootstrap sampling to calculate proportions.

    :param data: pandas Series of binary values (0 or 1).
    :param n_bootstrap: Number of bootstrap samples.
    :param random_state: Seed for reproducibility.
    :return: List of bootstrap proportion estimates.
    """
    np.random.seed(random_state)
    proportions = []
    for _ in range(n_bootstrap):
        sample = data.sample(n=len(data), replace=True)
        proportion = sample.sum() / len(sample)
        proportions.append(proportion)
    return proportions


def plot_relevancy_bars(
    df,
    selected_relevance_type,
    filter_understanding,
    understanding_column,
    output_filepath,
    dataset_mapping,
    colors,
    title,
) -> None:
    """Create a plot of relevancy scores for models and datasets, and save it to the specified file.

    :param df: pandas DataFrame containing the data.
    :param selected_relevance_type: Column name of the relevancy metric to plot.
    :param filter_understanding: Whether to filter rows based on understanding.
    :param understanding_column: Column used for filtering (if filter_understanding is True).
    :param output_filepath: Filepath to save the generated plot.
    :param dataset_mapping: Mapping dictionary to adjust dataset and modality labels.
    """
    unique_models = df["model"].unique()
    unique_datasets = df["dataset"].unique()
    num_models = len(unique_models)
    num_datasets = len(unique_datasets)

    # Define custom hex colors (ensure consistency)
    # colors = ["#1f77b4", "#ff7f0e"]  # Blue for 'txt', Orange for 'img'

    # Create figure and GridSpec for better layout control
    fig = plt.figure(figsize=(5 * num_datasets + 2, 4 * num_models))
    gs = GridSpec(
        num_models + 1,
        num_datasets + 2,
        width_ratios=[0.12, 1, 0.25] + [1] * (num_datasets - 1),
        height_ratios=[0.12] + [1] * num_models,
        wspace=0.1,
        hspace=0.3,
    )

    for row_idx, model in enumerate(unique_models):
        for col_idx, dataset in enumerate(unique_datasets):
            if dataset == "vicunabench":
                ax = fig.add_subplot(
                    gs[row_idx + 1, col_idx + 1]
                )  # +1 to skip the first row for dataset headers and first column for model labels
            else:
                ax = fig.add_subplot(
                    gs[row_idx + 1, col_idx + 2]
                )  # Adjust for dashed line after vicunabench

            # Filter data for the current model and dataset
            df_model_dataset = df[(df["model"] == model) & (df["dataset"] == dataset)]

            if filter_understanding and understanding_column:
                df_filtered = df_model_dataset[
                    df_model_dataset[understanding_column] == 1
                ]
            else:
                df_filtered = df_model_dataset

            if df_filtered.empty:
                print(
                    f"No data available for Model '{model}' and Dataset '{dataset}' after filtering."
                )
                # Indicate no data in the subplot
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                    transform=ax.transAxes,
                )
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                continue

            # Extract relevance_type scores for 'txt' and 'img'
            txt_scores = df_filtered[df_filtered["modality"] == "txt"][
                selected_relevance_type
            ]
            img_scores = df_filtered[df_filtered["modality"] == "img"][
                selected_relevance_type
            ]

            # Perform bootstrap sampling to calculate proportions
            n_bootstrap = 100
            bootstrap_txt = bootstrap_proportions(
                txt_scores, n_bootstrap=n_bootstrap, random_state=42
            )
            bootstrap_img = bootstrap_proportions(
                img_scores, n_bootstrap=n_bootstrap, random_state=42
            )

            # Calculate the median proportion for each modality
            proportion_txt = np.median(bootstrap_txt)
            proportion_img = np.median(bootstrap_img)

            # Calculate the 5th and 95th percentiles for error bars
            lower_txt = np.percentile(bootstrap_txt, 5)
            upper_txt = np.percentile(bootstrap_txt, 95)
            lower_img = np.percentile(bootstrap_img, 5)
            upper_img = np.percentile(bootstrap_img, 95)

            # Calculate error ranges
            error_txt = [
                proportion_txt - lower_txt,
                upper_txt - proportion_txt,
            ]
            error_img = [
                proportion_img - lower_img,
                upper_img - proportion_img,
            ]

            # Prepare data for plotting
            modalities = ["txt", "img"]
            proportions = [proportion_txt, proportion_img]
            errors = [error_txt, error_img]

            # Define positions and width for bars, closer together
            bar_width = 0.3  # Adjusted to bring bars closer
            x_positions = np.array([0.2, 0.8])  # Custom positions to reduce space

            # Prepare yerr in the required format
            lower_errors, upper_errors = zip(*errors)
            yerr = [lower_errors, upper_errors]  # Shape should be (2,2)

            # Map modality labels based on the JSON file (replace "txt" and "img" with the mapped values)
            modalities_mapped = [dataset_mapping.get(mod, mod) for mod in modalities]

            # Plot bar plots with error bars
            bars = ax.bar(
                x_positions,
                proportions,
                yerr=yerr,
                align="center",
                alpha=0.7,
                ecolor="black",
                capsize=5,
                color=colors,
                width=bar_width,
            )

            # Set x-axis labels with mapped values (e.g., "text only" and "multimodal")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                modalities_mapped, fontsize=23
            )  # Increased font size for labels

            # Retain y-axis ticks and numbers
            ax.set_yticks(np.linspace(0, 1, 5))
            ax.set_ylim(0, 1)
            ax.tick_params(axis="y", labelsize=16)

            # Remove individual subplot titles
            ax.set_title("")

            # Add grid for better readability
            ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Add dataset names as column headers
    for col_idx, dataset in enumerate(unique_datasets):
        if dataset == "vicunabench":
            ax = fig.add_subplot(
                gs[0, col_idx + 1]
            )  # +1 to skip the first row for dataset headers and first column for model labels
        else:
            ax = fig.add_subplot(
                gs[0, col_idx + 2]
            )  # Adjust for dashed line after vicunabench
        # Hide the axes
        ax.axis("off")

        # Use the dataset mapping from JSON, if available, else use the original
        dataset_name = dataset_mapping.get(dataset, dataset)

        # Add dataset name text
        ax.text(
            0.5,
            0.5,
            dataset_name,
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
        )  # Increased font size

    # Add model names as row labels in the first column
    for row_idx, model in enumerate(unique_models):
        ax = fig.add_subplot(gs[row_idx + 1, 0])
        # Hide the axes
        ax.axis("off")
        # Add model name text, horizontal and closer to the edge
        ax.text(
            0.9,
            0.5,
            model,
            ha="right",
            va="center",
            fontsize=26,
            fontweight="bold",
            rotation=0,
            transform=ax.transAxes,
        )  # Increased font size

    # Add a dashed line to separate vicunabench from other datasets
    ax = plt.subplot(gs[:, 2])
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_axis_off()

    # Adjust layout to reduce white space around edges and bring bars together
    plt.suptitle(title, fontsize=32, y=1.04)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Save the plot
    # plt.tight_layout()
    plt.savefig(output_filepath, bbox_inches="tight", pad_inches=0.3)
    plt.close()
    print(f"Plot saved as '{output_filepath}'.")
