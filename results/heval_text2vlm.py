# Copyright (c) 2025 Advai Ltd.
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""
Streamlit app for reviewing mutated prompts and their summaries.
"""
import json
import random
import re
from typing import Any

import streamlit as st
from PIL import Image


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


def highlight_text(text, salient_words, color="orange"):
    # Highlight tags <>
    text = re.sub(r"(<[^>]*>)", r'<span style="color:red">\1</span>', text)

    # Highlight salient words
    for word in salient_words:
        # Escape any special characters in the word
        escaped_word = re.escape(word)
        # Create a regular expression pattern to match the word
        pattern = rf"\b{escaped_word}\b"
        text = re.sub(
            pattern,
            f'<span style="color:{color}">{word}</span>',
            text,
            flags=re.IGNORECASE,
        )

    return text


# Main Streamlit app
def main(config) -> None:
    st.title("Prompt Reviewer")

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
    mutated_prompt = current_entry["mutated_prompt"]
    summary = current_entry["summary"]
    tagged_prompt = current_entry["tagged_prompt"]
    salient_words_path = current_entry["salient_words"]
    salient_image_path = current_entry["image_of_salient_words"]

    # Load salient words from file
    if f"salient_words_{current_index}" not in st.session_state:
        with open(salient_words_path, "r") as f:
            st.session_state[f"salient_words_{current_index}"] = f.read().splitlines()

    salient_words = st.session_state[f"salient_words_{current_index}"]

    st.subheader("Mutated Prompt")
    st.write(mutated_prompt)

    st.subheader("Summary")
    st.write(summary)

    # Retrieve existing review data or use default values
    review_data = st.session_state.review_data.get(
        current_index,
        {
            "summary_review": "Good",
            "summary_comment": "",
            "all_extracted": False,
            "all_valid": False,
        },
    )

    # User input for reviewing the summary
    summary_review = st.radio(
        "Review the Summary",
        ["Very Bad", "Bad", "Good", "Great"],
        key=f"summary_review_{current_index}",
        index=["Very Bad", "Bad", "Good", "Great"].index(review_data["summary_review"]),
    )
    summary_comment = st.text_area(
        "Additional Comments on the Summary",
        key=f"summary_comment_{current_index}",
        value=review_data["summary_comment"],
    )

    st.subheader("Salient Words")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(highlight_text(summary, salient_words), unsafe_allow_html=True)
    with col2:
        st.markdown(
            highlight_text(tagged_prompt, salient_words),
            unsafe_allow_html=True,
        )
        st.write(Image.open(salient_image_path))

    # User input for reviewing the salient words
    all_extracted = st.checkbox(
        "All salient words extracted?",
        key=f"all_extracted_{current_index}",
        value=review_data["all_extracted"],
    )
    all_valid = st.checkbox(
        "All salient words valid?",
        key=f"all_valid_{current_index}",
        value=review_data["all_valid"],
    )

    # Callback function to save current feedback and move to the next entry
    def save_and_next():
        # Save current feedback
        st.session_state.review_data[current_index] = {
            "summary_review": summary_review,
            "summary_comment": summary_comment,
            "all_extracted": all_extracted,
            "all_valid": all_valid,
        }
        st.session_state.results[current_index] = {
            "mutated_prompt": mutated_prompt,
            "summary": summary,
            "salient_words": salient_words,
            "summary_review": summary_review,
            "summary_comment": summary_comment,
            "all_extracted": all_extracted,
            "all_valid": all_valid,
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
    def go_back():
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
        "data_path": "mutated_prompts/final_result_interpreter.json",
        "num_to_review": 2,
        "save_path": "reviewed/text2vlm/review_results_interpreter.json",
    }

    main(config)
