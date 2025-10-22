# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
Streamlit app for reviewing classifier outputs.

This app allows users to review a random selection of entries from a dataset,
providing feedback on relevancy scores and classification correctness.
"""
import json
import random
from typing import Any

import streamlit as st


def map_rlhf_refusal(value: int) -> str:
    """Map the integer value of 'rlhf_refusal' to a human-readable string."""
    rlhf_mapping: dict[int, str] = {
        0: "Non Refusal",
        1: "Refusal",
    }
    return rlhf_mapping.get(value, "Unknown")


def load_data(file_path):
    """Load your dataset (assumed to be a JSON file)."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def get_random_entries(data, n) -> list[Any]:
    """Select n random entries from the dataset."""
    return random.sample(data, n)


def save_results(results, output_path) -> None:
    """Save the results to a new JSON file."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)


# Main Streamlit app
def main(config) -> None:
    st.title("Classifier Reviewer")

    data = load_data(config["data_path"])

    # Initialize session state for random entries and current index
    if "random_entries" not in st.session_state:
        n = config["num_to_review"]  # Number of entries to review
        st.session_state.random_entries = get_random_entries(data, n)
        st.session_state.current_index = 0
        st.session_state.results = [None] * n  # Placeholder for results
        st.session_state.review_data = {}  # Track review data for each entry

    current_index = st.session_state.current_index
    current_entry = st.session_state.random_entries[current_index]
    summary = current_entry["summary"]
    response = current_entry["response"]
    relevancy = current_entry["relevancy"]
    rlhf_refusal = current_entry["rlhf_refusal"]

    st.subheader("Summary")
    st.write(summary)

    st.subheader("Response")
    st.write(response)

    # Retrieve existing review data or use default values
    review_data = st.session_state.review_data.get(
        current_index,
        {
            "relevancy_review": "Good",
            "relevancy_comment": "",
            "correct_classification": False,
        },
    )

    st.subheader("Relevancy Scores")
    st.write(relevancy)

    # User input for reviewing the relevancy
    relevancy_review = st.radio(
        "Review the Relevancy Scores",
        ["Very Bad", "Bad", "Good", "Great"],
        key=f"relevancy_review_{current_index}",
        index=["Very Bad", "Bad", "Good", "Great"].index(
            review_data["relevancy_review"]
        ),
    )
    relevancy_comment = st.text_area(
        "Additional Comments on the relevancy",
        key=f"relevancy_comment_{current_index}",
        value=review_data["relevancy_comment"],
    )

    st.subheader("Classification")
    st.write(map_rlhf_refusal(rlhf_refusal))

    # User input for reviewing the salient words
    correct_classification: bool = st.toggle(
        "Correct Classification?",
        key=f"correct_classification_{current_index}",
        value=review_data["correct_classification"],
    )

    # Callback function to save current feedback and move to the next entry
    def save_and_next() -> None:
        # Save current feedback
        st.session_state.review_data[current_index] = {
            "relevancy_review": relevancy_review,
            "relevancy_comment": relevancy_comment,
            "correct_classification": correct_classification,
        }
        st.session_state.results[current_index] = {
            "summary": summary,
            "response": response,
            "relevancy": relevancy,
            "classification": rlhf_refusal,
            "relevancy_review": relevancy_review,
            "relevancy_comment": relevancy_comment,
            "correct_classification": correct_classification,
        }

        # Move to the next entry
        if current_index < len(st.session_state.random_entries) - 1:
            st.session_state.current_index += 1
        else:
            st.success("All entries reviewed!")
            save_path = config["save_path"]
            save_results(st.session_state.results, save_path)
            st.write(f"Results saved to {save_path}")

    # Callback function to go back to the previous entry
    def go_back() -> None:
        if current_index > 0:
            st.session_state.current_index -= 1

    # Save & Next button
    st.button("Save & Next", on_click=save_and_next)

    # Previous button
    st.button("Previous", on_click=go_back)

    # Display the current index and total count
    st.write(f"Entry {current_index + 1} of {len(st.session_state.random_entries)}")


if __name__ == "__main__":
    # Edit paths as needed
    config = {
        "data_path": "responses_img.json",
        "num_to_review": 1,
        "save_path": "reviewed/classifier/review_results_toxi_vila40b_txt.json",
    }

    main(config)
