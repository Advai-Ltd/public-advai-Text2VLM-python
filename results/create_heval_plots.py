# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""Generate combined plots for reviewed Text2VLM and Classifier results.
This script processes JSON files containing reviewed results for both Text2VLM
summaries and classifier outputs. It aggregates the review data, computes
statistics, and generates combined plots to visualize the performance of the
Text2VLM model and classifiers based on human reviews."""
import json
import os
from collections import Counter
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

X_LABEL_COORD: float = -0.2


def load_json_data(
    file_list: List[str], keys: List[str]
) -> Tuple[int, List[List[Any]]]:
    """Load JSON data from a list of files and extracts specified fields."""
    combined_data = {key: [] for key in keys}
    data_size: int = 0

    for file_name in file_list:
        with open(file_name, "r") as f:
            data = json.load(f)
            data_size += len(data)

        for key in keys:
            combined_data[key].extend([item[key] for item in data])

    return data_size, [combined_data[key] for key in keys]


def process_json_text2vlm(
    file_list: List[str],
):
    """Process JSON files to extract 'summary_review', 'all_extracted', and 'all_valid' fields."""
    keys = ["summary_review", "all_extracted", "all_valid"]
    return load_json_data(file_list, keys)


def process_json_classifier(
    file_list: List[str],
):
    """Process JSON files to extract 'relevancy_review' and 'correct_classification' fields."""
    keys = ["relevancy_review", "correct_classification"]
    return load_json_data(file_list, keys)


def plot_combined_results(
    summary_counts,
    extracted_count,
    valid_count,
    relevancy_counts,
    classifier_count,
    text2vlm_size,
    classifier_size,
    file_name: str,
):
    """Plot combined results in a single figure with two subplots."""
    # Labels for text2vlm plot
    labels_summary: List[str] = ["Great", "Good", "Bad", "Very Bad"]
    labels_extraction: List[str] = ["All Extracted", "All Valid"]
    labels_text2vlm = labels_summary + labels_extraction

    # Labels for classifier plot
    labels_relevancy = ["Great", "Good", "Bad", "Very Bad"]
    labels_classifier = ["Correct"]

    # Values for text2vlm
    summary_values = [summary_counts.get(label, 0) for label in labels_summary]
    summary_values.extend([extracted_count, valid_count])

    # Values for classifier
    classifier_values = [
        relevancy_counts.get(label, 0) for label in labels_relevancy
    ]
    classifier_values.append(classifier_count)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Values for text2vlm (convert counts to percentages)
    summary_values = [
        (summary_counts.get(label, 0) / text2vlm_size) * 100
        for label in labels_summary
    ]
    summary_values.extend(
        [
            (extracted_count / text2vlm_size) * 100,
            (valid_count / text2vlm_size) * 100,
        ]
    )

    # Values for classifier (convert counts to percentages)
    classifier_values = [
        (relevancy_counts.get(label, 0) / classifier_size) * 100
        for label in labels_relevancy
    ]
    classifier_values.append((classifier_count / classifier_size) * 100)

    # Text2VLM Plot
    x_text2vlm = np.arange(len(labels_text2vlm))
    ax1.bar(
        x_text2vlm,
        summary_values,
        color=["skyblue"] * len(labels_summary)
        + ["orange"] * len(labels_extraction),
    )
    ax1.set_xlabel("Summary Reviews and Extraction Reviews")
    ax1.xaxis.set_label_coords(0.5, X_LABEL_COORD)
    ax1.set_ylabel("Percentage")
    ax1.set_title("Reviewed Text2VLM Results")
    ax1.set_xticks(x_text2vlm)
    ax1.set_xticklabels(labels_text2vlm, rotation=45)
    ax1.axvline(
        len(labels_summary) - 0.5, color="gray", linestyle="--"
    )  # Separator line
    ax1.set_ylim(0, 100)
    # Create proxy artists for the legend in the Text2VLM plot
    skyblue_patch = Line2D([0], [0], color="skyblue", lw=4, label="Summarizer")
    orange_patch = Line2D([0], [0], color="orange", lw=4, label="Extractor")
    ax1.legend(handles=[skyblue_patch, orange_patch])

    # Classifier Plot
    x_classifier = np.arange(len(labels_relevancy) + len(labels_classifier))
    ax2.bar(
        x_classifier,
        classifier_values,
        color=["#2ca02c"] * len(labels_relevancy)
        + ["#9467bd"] * len(labels_classifier),
    )

    ax2.set_xlabel("Relevancy Reviews and Refusal Reviews")
    ax2.xaxis.set_label_coords(0.5, X_LABEL_COORD)
    ax2.set_ylabel("Percentage")
    ax2.set_title("Reviewed Classifier Results")
    ax2.set_xticks(x_classifier)
    ax2.set_xticklabels(labels_relevancy + labels_classifier, rotation=45)
    ax2.axvline(
        len(labels_relevancy) - 0.5, color="gray", linestyle="--"
    )  # Separator line
    ax2.set_ylim(0, 100)
    # Create proxy artists for the legend in the Classifier plot
    green_patch = Line2D(
        [0], [0], color="#2ca02c", lw=4, label="Relevancy Classifier"
    )
    purple_patch = Line2D(
        [0], [0], color="#9467bd", lw=4, label="RLHF Refusal Classifier"
    )
    ax2.legend(handles=[green_patch, purple_patch])

    plt.tight_layout()
    plt.savefig(f"plots/{file_name}.pdf")
    plt.close()


def main() -> None:
    # Process Text2VLM data
    directory_text2vlm = "reviewed/text2vlm/"
    file_list_text2vlm = [
        f"{directory_text2vlm}{f}"
        for f in os.listdir(directory_text2vlm)
        if f.endswith(".json")
    ]
    (
        data_size_text2vlm,
        (combined_reviews, combined_extracted, combined_valid),
    ) = process_json_text2vlm(file_list_text2vlm)
    summary_counts = Counter(combined_reviews)
    extracted_count = combined_extracted.count(True)
    valid_count = combined_valid.count(True)

    # Process Classifier data
    directory_classifiers = "reviewed/classifier/"
    file_list_classifiers = [
        f"{directory_classifiers}{f}"
        for f in os.listdir(directory_classifiers)
        if f.endswith(".json")
    ]
    (
        data_size_classifier,
        (combined_reviews_classifiers, combined_correct_classification),
    ) = process_json_classifier(file_list_classifiers)
    relevancy_counts = Counter(combined_reviews_classifiers)
    classifier_count = combined_correct_classification.count(True)

    # Plot combined results
    plot_combined_results(
        summary_counts,
        extracted_count,
        valid_count,
        relevancy_counts,
        classifier_count,
        data_size_text2vlm,
        data_size_classifier,
        "combined_plot",
    )


if __name__ == "__main__":
    main()
