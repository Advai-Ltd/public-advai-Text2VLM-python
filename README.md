# Text2VLM

[![arXiv](https://img.shields.io/badge/arXiv-2507.20704-b31b1b.svg)](https://arxiv.org/abs/2507.20704)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**Generated responses and data have been excluded from the repository as they can contain harmful/malicious information.**

## Overview

`Text2VLM` is a CLI tool to transform textual data used in Large Language Model (LLM) evaluations into a format suitable for Visual Language Models (VLMs). The service processes data by summarizing prompts, extracting salient words, generating images of those words and tagging the original text with references to the images. The final output is a json file of text prompt and a .data file containing the images and cached data.

The `TargetModel` option runs a model from `Replicate` or a local `Vila` model (must be running the a vila server) and then runs a relevancy classifier on the responses. The `TargetModel` will iterate through the generated dataset and create a new `responses.json` containing the targets models responses and the relevancy scores.

The final CLI option is `Classify`, it iterates through all the reponses and classifys whether they contain a RLHF refusal or not.

Additionally, `heval_text2vlm` and `heval_classifiers` provides a Streamlit GUI to allow for human review of some of the steps in the `Text2VLM`, the relevancy and the rlhf refusal classifiers. It takes the output from the service, randomly selects N number of samples and lets the user rate and make comments. The results are then saved as a json.

![Pipeline](assets/methodology_pipeline.png "Text2VLM Pipeline")

## Features

### Command-Line Interface (CLI)

The repository includes an interactive Command-Line Interface (CLI) tool to perform various functionalities such as converting text datasets into Visual Language Model (VLM) format, running target models on input data, and classifying responses. The tool is designed to be user-friendly, guiding users through available options.

### Text2VLM Features

- **Prompt Summarization**: Summarizes text prompts using the dolphin model.
- **Salient Word Extraction**: Extracts critical words or phrases using 4o-mini that are essential to the text's meaning.
- **Image Generation**: Creates images representing the extracted salient words.
- **Tagging**: Tags the original prompts with references to the generated images.
- **Caching**: Caches intermediate results to avoid repeated API calls.

### TargetModel

- **VILA 1.5**: Allows for the use of VILA models through a local server
- **Replicate**: Allows for the use of replicate models
- **Relevancy Classification**: Calls the relevancy classifier to classify the VLMs response.
- **Save & Reload**: Saves all responses to a JSON file, also can reload previous responses and continue through the rest of the dataset.

### Classify

- **RLHF Classification**: Classifies the responses as a RLHF refusal or not.

### Results

- **Human Reviews (Pipeline or Classifiers)**
    1. **Random Prompt Selection**: Randomly selects a specified number of prompts from the dataset for review.
    2. **Review**: Users can provide feedback on the pipeline and classifiers.
    3. **Save and Continue**: Progress is saved as you review each prompt, and you can navigate between prompts using "Previous" and "Save & Next" buttons.
    4. **Save to JSON**: Results are then saved to a JSON file.

- **Plots**
    1. **Human Evaluations**: Creates a figure with 2 subplots containing the results of the Human Reviews.
    2. **Results**: Creates 2 figures showing the refusals and model understanding across datasets and models.

## Installation

### Requirements
- Python 3.10 or higher

### Steps

1. **Clone the repository**

2. **Install the package and all dependencies**:

    ```bash
    pip install -e .
    ```
    
    This will automatically install all required dependencies from `requirements.txt`.

3. **Set your API keys**:
    - **OpenAI API key**: Ensure that your OpenAI API key is set in your environment variables. Add the following line to your `.bashrc` file:

        ```bash
        export OPENAI_API_KEY="<INSERT KEY>"
        ```

    - **Replicate API key**: Ensure that your Replicate API key is set in your environment variables. Add the following line to your `.bashrc` file:

        ```bash
        export REPLICATE_API_TOKEN="<INSERT KEY>"
        ```

    Then, reload your `.bashrc` file:

    ```bash
    source ~/.bashrc
    ```

## Data Format

The Text2VLM service can take 3 file formats, .txt, .csv and .json.

- **.txt and .csv**: New line for each new prompt.
- **.json**:  List of jsons with a mutated_prompt and attack_type field.

```json
[
    {
        "mutated_prompt": "",
        "attack_type": "",
    }
]
```

## Usage

### CLI

To start the CLI, ensure that you have installed the required dependencies (also start the vila server if required) and set the necessary environment variables. Then, run the CLI tool using the following command:

`python run_cli.py`

This will launch an interactive menu with the following options:

#### 1. Text2VLM

Converts text datasets into VLM format, providing the user with a selection of available datasets and options to specify the number of samples to process.

- **Interactive Steps:**
  - Select a dataset from the available options or provide a filepath.
  - Specify the number of samples to process (-1 to process all).
  - The script will validate your inputs and provide feedback in case of errors.

#### 2. Run Target Model

Allows you to run a predefined model on the given input data. Models are available from the **Replicate** API and **VILA models**, and you can select one to run.

- **Interactive Steps:**
  - Choose from a list of models (e.g., `Llava-7b`, `VILA 1.5`).
  - Input the data you want to process with the model.
  - The script will execute the model and return the result, handling errors and validation along the way.

#### 3. Classify

Classifies responses using a predefined model. This option allows you to classify a dataset's responses as an **RLHF Refusal** or not.

- **Interactive Steps:**
  - Provide the path to the dataset file containing the responses to classify.
  - The script validates the file path and type, ensuring it points to a valid `.json` file.

#### Adding new Replicate models to TargetModel

Adding a new replicate model to `TargetModel` requires changes in a few areas.

1. **Replicate Model Path**: Add the replicate model to the MODEL_MAPPING in cli.py, requires a shorthand name and the model path (e.g: "Llava-34b": "yorickvp/llava-v1.6-34b:41ecfbfb261e6c1adf3ad896c9066ca98346996d7c4045c5bc944a79d430f174").

2. **Input Dictionary Parameters**: Some replicate models take/have unique parameter names. To change this, go into and change the `input` dictionary in the `perform_inference()` function found in `target_model.py`.

## Reproduce Results

### Text2VLM Human Review Example

Make sure to run the Text2VLM service in the CLI before running the Human Review. It will require a generated `final_results` json and a corresponding `.data` directory.

1. **Create output directories**:
    ```bash
    mkdir -p reviewed/text2vlm
    ```

2. **Run the Application**
    Edit the `heval_text2vlm.py` config.
    Then to start the Streamlit app, run:
        ```
            python -m streamlit run results/heval_text2vlm.py
        ```

3. **Review Prompts**
    - The application will display a prompt, summary, and salient words for each entry.
    - Provide your feedback using the radio buttons, text area, and checkboxes.
    - Click "Save & Next" to save your review and move to the next prompt.
    - Click "Previous" to navigate back to the previous prompt.

4. **Results Storage**
    - After reviewing ALL prompts, the results are saved to a JSON file named review_results.json.

5. **Config Customization**
    - **Number of Prompts:** Modify the number of prompts to review by changing the n variable in the main() function.
    - **Dataset Path:** Update the dataset_path variable in the main() function to point to your dataset.
    - **Save Path** Update the filepath to save the results of the review.

### Relevancy and RLHF Refusal Human Review Example

Make sure to run all options in the CLI before running the Human Review. It is required to have a response json containing the relevancy scores and the rlhf classifications.

1. **Create output directories**:
    ```bash
    mkdir -p reviewed/classifier
    ```

2. **Run the Application**
    Edit the `heval_classifiers.py` config.
    Then to start the Streamlit app, run:
        ```
            python -m streamlit run results/heval_classifiers.py
        ```

3. **Review Relevancy Scores and Refusal Classification**
    - The application will display the summary/prompt, the target models response, the relevancy results and the rlhf classification.
    - Provide your feedback using the radio buttons, text area, and toggle.
    - Click "Save & Next" to save your review and move to the next prompt.
    - Click "Previous" to navigate back to the previous prompt.

4. **Results Storage**
    - After reviewing ALL prompts, the results are saved to a JSON file named review_results.json.

5. **Config Customization**
    - **Number of Prompts:**: Modify the number of prompts to review by changing the n variable in the main() function.
    - **Dataset Path:**: Update the dataset_path variable in the main() function to point to your dataset.
    - **Save Path**: Update the filepath to save the results of the review.

### Plots

1. **Create output directories**:
    ```bash
    mkdir -p plots
    ```

2. **Human Review**: Requires the review of both the pipeline and classifiers to be complete with the relevant jsons saved under `reviewed/text2vlm/` and `reviewed/classifier/`. The plot is then saved as `plots/combined_plot.pdf`.

3. **Results**: Will generate 2 figures with 9 subplots each. First run `results/get_scores_dataframe.py` to generate `results/relevancy_scores.tsv`. Once the tsv has been created, run `results/plot_results.py`. Both figures will be saved in `plots/`.
