# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
Generate plots for relevancy scores from TSV data.
"""
import json
import os

import pandas as pd
from plot_utils import plot_relevancy_bars

# Define the path to the TSV file
tsv_file_path = "results/relevancy_scores.tsv"
# Define the path to the JSON mapping file
json_mapping_path = "results/label2plot_mappings.json"

# Load the DataFrame from the TSV file
df = pd.read_csv(tsv_file_path, sep="\t")

# Load the JSON mapping file
try:
    with open(json_mapping_path, "r") as f:
        dataset_mapping = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find the dataset mapping file at {json_mapping_path}.")
    dataset_mapping = {}

# Directory to save the plots
dir_plots = "plots/"

# Ensure the directory exists
if not os.path.exists(dir_plots):
    os.makedirs(dir_plots)
    print(f"Directory created: {dir_plots}")

# Set of variables for different configurations
plot_configs = [
    (
        "rlhf_refusal",
        True,
        "understanding_Accuracy/Correctness",
        "Safety Alignment: Model Refusal Rates Across Datasets",
    ),
    (
        "understanding_Accuracy/Correctness",
        False,
        None,
        "Target Model Understanding Scores for Each Dataset",
    ),
]


# Function to generate filenames for plots
def generate_filename(relevance_type, filter_understanding, understanding_column):
    filename = f"{relevance_type}_filter_{filter_understanding}"
    if understanding_column:
        filename += f"_{understanding_column}"

    filename = filename.replace("/", "_")

    return f"{filename}.pdf"


colors: list[list[str]] = [
    [
        "#1f77b4",  # Blue for 'txt'
        "#ff7f0e",  # Orange for 'img'
    ],
    [
        "#2ca02c",  # Green for 'txt'
        "#9467bd",  # Purple for 'img'
    ],
]
# Loop over configurations and generate plots
for i, (
    relevance_type,
    filter_understanding,
    understanding_column,
    title,
) in enumerate(plot_configs):
    filename = generate_filename(
        relevance_type, filter_understanding, understanding_column
    )
    output_filepath = os.path.join(dir_plots, filename)

    # Alternate between two color schemes
    current_colors = colors[i % 2]

    # Call the plotting function
    plot_relevancy_bars(
        df,
        relevance_type,
        filter_understanding,
        understanding_column,
        output_filepath,
        dataset_mapping,
        current_colors,
        title,
    )

print("Plots generated successfully!")
