from setuptools import find_packages, setup

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="text2vlm",
    version="1.0.0",
    author="Advai Ltd",
    author_email="contact@advai.ltd",  # Update with actual email
    description="CLI tool to transform textual data for Visual Language Models (VLMs)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Advai-Ltd/Text2VLM",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "text2vlm=text2vlm.text_to_multimodal.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "text2vlm": [
            "text_to_multimodal/prompts/*.txt",
            "datasets/*.json",
            "datasets/medsafety_datasets/**/*.csv",
            "datasets/toxigen_prompts/*.txt",
        ],
    },
)
