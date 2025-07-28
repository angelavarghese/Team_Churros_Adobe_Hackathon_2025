
# README.md (Comprehensive Guide)

```markdown
# Team Churros - Adobe Hackathon 2025: Persona-Driven Document Intelligence

This repository contains the solution for "Persona-Driven Document Intelligence" (Challenge 1B) and integrates with a teammate's PDF processing application (Challenge 1A).

## Challenge Overview

This project builds an intelligent system that acts as a document analyst. Given a collection of PDF documents, a specific user persona, and a "job-to-be-done," the system extracts and prioritizes the most relevant sections. It provides both a command-line interface (CLI) for batch processing and a user-friendly Flask web application for interactive use.

Key Features:

* Persona-Driven Relevance: Identifies content based on the user's role and specific task.
* Intelligent Document Analysis: Extracts key sections and refined summaries.
* Multilingual Support: Processes documents in multiple languages (e.g., English, Japanese) using multilingual embeddings.
* Advanced PDF Processing (Integrated from Challenge 1A):
    * Detects PDF type (text-based vs. scanned).
    * Performs OCR (using Tesseract) for scanned PDFs.
    * Extr`acts` rich features (font size, alignment, spacing) from text blocks.
    * Classifies headings using a trained Machine Learning model (RandomForest) for accurate section identification.
* Smart Content Filtering: Excludes irrelevant or undesirable content (e.g., "meat" in a "vegetarian" context).
* Containerized (Docker): Ensures a reproducible and isolated environment for easy setup and deployment.

## Prerequisites

Before you begin, ensure you have the following installed:

* Git: For cloning the repository.
* Docker Desktop: (For Windows/macOS) Provides the Docker engine and WSL2 integration.
* Docker Engine: (For Linux)
* Python 3.12 or higher: It's recommended to use a virtual environment.

### Docker Desktop Resources (Crucial for Building)

The ML models (SentenceTransformers, RandomForest) and OCR tools (Tesseract, Poppler) are memory-intensive during installation. To avoid `SIGBUS` errors during Docker builds, ensure Docker Desktop is allocated sufficient resources:

1.  Open Docker Desktop Settings.
2.  Go to Resources (or Advanced).
3.  Set Memory to at least 6GB - 8GB (or more, if your system has spare RAM).
4.  Set CPUs to at least 4.
5.  Click Apply & Restart.

## Setup Instructions

1.  Clone the Repository:
    ```bash
    git clone [https://github.com/your_github_username/Team_Churros_Adobe_Hackathon_2025.git](https://github.com/your_github_username/Team_Churros_Adobe_Hackathon_2025.git)
    cd Team_Churros_Adobe_Hackathon_2025/Challenge_1B/
    ```

2.  Create Python Virtual Environment (Local Development):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows CMD: .\venv\Scripts\activate
    ```

3.  Install Python Dependencies (Local Development - for basic local testing if needed):
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This `venv` is for local development/testing. Docker builds its own environment.*

4.  Prepare Files (Challenge 1A Integration):
    * Ensure the `Challenge_1A/` folder exists at the root of the `Team_Churros_Adobe_Hackathon_2025/` repository.
    * Make sure `Challenge_1A/` contains `robust_pdf_extractor.py`, `predict_and_export.py`, `extract_features.py`, and the `model/` subfolder (with `model.joblib` and `label_map.json`).
    * Crucially, create an empty `__init__.py` file inside `Challenge_1A/` to make it a Python package:
        ```bash
        touch ../Challenge_1A/__init__.py
        ```

## Input Format

Your input data should be organized into a "collection" folder with a specific structure:

````

MyCollection/
├── PDFs/
│   └── document1.pdf
│   └── document2.pdf
└── input.json

````

The `input.json` file defines the documents to be processed, the user persona, and the job-to-be-done.

Example `input.json` Format:

```json
{
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Title of Document 1"
        },
        {
            "filename": "document2.pdf",
            "title": "Title of Document 2"
        }
    ],
    "persona": {
        "role": "Comparative Religious Scholar"
    },
    "job_to_be_done": {
        "task": "Identify and summarize key philosophical concepts related to Dharma, Karma, and self-realization from the text."
    }
}
````

## Running the Application

There are two primary ways to run the application:

### Option 1: CLI (Command-Line Interface) with Docker

This option runs the `pdf_processor.py` script directly within a Docker container for a single batch process.

1.  Build the CLI Docker Image:
    Navigate to the `Challenge_1B/` directory:

    ```bash
    cd ~/Projects/Team_Churros_Adobe_Hackathon_2025/ 
    docker build -t persona-doc-intel-flask -f Dockerfile_flask .
    ```

    *(Note: This uses `Dockerfile` (without `_flask`) to build the CLI-specific image.)*

2.  Prepare your Input Collection:
    Place your `MyCollection/` folder (containing `PDFs/` and `input.json`) within your `Challenge_1B/` directory.

3.  Run the CLI Docker Container:
    Replace 'i' with the appropriate number.
    ```bash
    cd ~/Projects/Team_Churros_Adobe_Hackathon_2025/ # Go to the root
    docker run \
    -v "$(pwd)/Challenge_1B/Collection i:/app/current_collection" \
    persona-doc-intel-cli \
    python pdf_processor.py /app/current_collection
    ```

      * Replace `MyCollection` with the actual name of your collection folder.
      * The `output.json` file will be generated directly inside your `MyCollection/` folder on your host machine.
      * Any raw processing output will be displayed in your terminal.

### Option 2: Flask Web Application with Docker

This option runs a web server that allows you to upload your collection (as a ZIP file) via a browser.

1.  Build the Flask Docker Image:
    Navigate to the `Challenge_1B/` directory:

    ```bash
    cd ~/Projects/Team_Churros_Adobe_Hackathon_2025/ 
    docker build -t persona-doc-intel-flask -f Dockerfile_flask .
    ```

    *(Note: This uses `Dockerfile_flask` to build the web app image.)*

2.  Run the Flask App Docker Container:

    ```bash
    cd ~/Projects/Team_Churros_Adobe_Hackathon_2025/ # Go to the root
    docker run -p 5000:5000 \
        -v "$(pwd)/Challenge_1B/templates:/app/templates" \ # Important if templates are being modified during dev
        persona-doc-intel-flask
    ```

      * `-p 5000:5000` maps port 5000 from the container to port 5000 on your host machine.

3.  Access the Web Application:
    Open your web browser and go to: `http://localhost:5000`

4.  Upload Your Collection:

      * On the webpage, click "Select Collection ZIP File."
      * Important ZIP Format: Your ZIP file (`e.g., my_test_data.zip`) must contain a single top-level folder (e.g., `MyCollection/`), and inside that folder should be your `PDFs/` directory and `input.json`.
      * Click "Process Document Collection." The results will be displayed on the page, and `output.json` will be available for download.
      * You can also access the `Challenge_1B/flask-zip-files-input`folder to access the collections in the zip format.

## Output

The application generates an `output.json` file (and displays its `subsection_analysis` on the Flask webpage).

Example `output.json` Structure:

```json
{
    "metadata": {
        "input_documents": ["document1.pdf", "document2.pdf"],
        "persona": "Comparative Religious Scholar",
        "job_to_be_done": "Identify and summarize key philosophical concepts...",
        "processing_timestamp": "2025-07-28T15:30:00.000000"
    },
    "extracted_sections": [
        {
            "document": "document1.pdf",
            "section_title": "Section Title from Doc1",
            "importance_rank": 1,
            "page_number": 5
        },
        {
            "document": "document2.pdf",
            "section_title": "Section Title from Doc2",
            "importance_rank": 2,
            "page_number": 10
        }
    ],
    "subsection_analysis": [
        {
            "document": "document1.pdf",
            "refined_text": "This is a concise summary of the most relevant content from Document 1, based on the persona's job-to-be-done.",
            "page_number": 5
        },
        {
            "document": "document2.pdf",
            "refined_text": "Here is another refined summary from Document 2, focusing on the key aspects for the user.",
            "page_number": 10
        }
    ]
}
```

## Core Concepts & Explanation

This project combines several advanced techniques from Document AI and Natural Language Processing:

### 1\. Robust PDF Processing (from Challenge 1A Integration)

At the foundation, this module (provided by a teammate) intelligently extracts text from PDFs, even complex ones:

  * PDF Type Detection: Automatically determines if a PDF is text-based (has selectable text) or scanned (an image of text).
  * OCR (Optical Character Recognition): For scanned PDFs, it uses Tesseract OCR (with Poppler utilities for image conversion) to convert images of text into machine-readable text.
  * Feature Extraction: Extracts rich visual and textual features from each block of text, including:
      * Font size & ratio, bold status.
      * Alignment (left, center, right) & spacing above/below.
      * Word count & average word length.
      * Script Detection (Multilingual): Identifies the script type (e.g., Latin, Devanagari, Japanese, Korean) of text blocks, crucial for multilingual support.
      * Unicode normalization for consistent text handling.
  * Heading Classification (Machine Learning): Utilizes a pre-trained RandomForest Classifier (or a heuristic fallback if the model is not found/fails) to accurately identify and categorize headings (e.g., H1, H2, H3, Title) based on the extracted features. This is significantly more accurate than simple rule-based methods.

### 2\. Persona-Driven Document Intelligence (Challenge 1B Core Logic)

Once the robust PDF processing provides accurate text blocks and heading outlines, our core logic takes over:

  * Text Embedding / Vectorization:
      * Uses SentenceTransformer models (`paraphrase-multilingual-MiniLM-L12-v2`) to convert all text (the user's persona/job, and extracted document sections/sentences) into high-dimensional numerical vectors (embeddings).
      * This multilingual model allows semantic comparison between an English query and content in languages like Japanese or Hindi, capturing the meaning regardless of language.
  * Semantic Relevance Ranking:
      * Calculates Cosine Similarity between the embedding of the user's combined persona-job query and the embeddings of each document section.
      * Sections are then ranked by these similarity scores, prioritizing the most semantically relevant content.
  * Section Selection (One per PDF): From the overall pool of relevant sections, the system selects the single most relevant section from *each unique input PDF*. These form the `extracted_sections` in the output, globally ranked.
  * Smart Content Filtering (Exclusion Logic):
      * For specific contexts (e.g., "vegetarian" meals, "gluten-free" diets), the system identifies a list of exclusion keywords (e.g., "meat", "wheat").
      * During `subsection_analysis`, if any sentence within a candidate summary contains these exclusion keywords, that sentence is entirely discarded. If all sentences for a section are filtered out, no `refined_text` is generated for that section's entry in `subsection_analysis`.
  * Extractive Summarization:
      * From the sentences within the top-ranked sections (after filtering for exclusions), the most semantically similar sentences to the query are selected.
      * These selected sentences form the `refined_text` for the `subsection_analysis`, providing concise and relevant summaries.
  * Text Cleaning: A custom function ensures the `refined_text` is free of newlines and contains only alphanumeric characters and standard punctuation, making it clean for display.

### 3\. Containerization (Docker)

  * Reproducibility & Isolation: All components (Python versions, libraries, system dependencies, trained models) are bundled into self-contained Docker images. This guarantees that the application runs identically across any environment and avoids "it works on my machine" issues.
  * Simplified Deployment: Docker images can be easily deployed to various cloud platforms or on-premise servers.
  * Modular Design: Separate Dockerfiles are used for the CLI processing logic (`Dockerfile`) and the Flask web application (`Dockerfile_flask`), allowing for independent development and potential future scaling.

## Troubleshooting Tips

  * `SIGBUS: bus error` during `docker build`:
      * Cause: Insufficient RAM allocated to Docker Desktop during the installation of large ML libraries.
      * Fix: Increase Docker Desktop's Memory allocation (e.g., to 8GB or more) in its settings.
      * Also: Run `docker system prune --all --volumes -f` before rebuilding to clear corrupted caches.
  * `ModuleNotFoundError: No module named '...'`:
      * Cause: A Python file is missing, or its import path is incorrect, or an `__init__.py` file is missing in a package directory.
      * Fix:
          * Ensure all `.py` files are in their correct subfolders (e.g., `Challenge_1A/`).
          * Verify `__init__.py` exists in every folder that acts as a Python package (e.g., `Challenge_1A/__init__.py`).
          * Check `COPY` commands in Dockerfiles.
          * Verify `requirements.txt` includes all necessary libraries.
  * `re.error: bad escape \p`:
      * Cause: The standard `re` module is being used instead of the `regex` module for Unicode character properties. This is usually due to improper import aliasing or conflicting imports.
      * Fix: Ensure `import regex` is used, and explicitly call `regex.sub()` and `regex.search()`. For teammate's code, ensure `import regex as re` and their uses are consistent. Rebuild Docker image.
  * "Input file not found" / "PDF not found" (during `docker run`):
      * Cause: Incorrect volume mount (`-v` flag) or incorrect path argument to `pdf_processor.py`.
      * Fix: Double-check that the host path in `-v` exactly matches your local folder, and the path inside the container (e.g., `/app/current_collection`) is correctly passed to `pdf_processor.py`.
  * "Model not found. Using heuristic-based classification":
      * Cause: `model.joblib` or `label_map.json` not found by the teammate's app.
      * Fix: Ensure `model/` folder exists under `Challenge_1A/` and contains the model files. Verify `MODEL_PATH` and `LABEL_MAP_PATH` in `robust_pdf_extractor.py` (and `predict_and_export.py`) correctly construct absolute paths using `os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "model.joblib")`. Rebuild Docker image.
  * Processing Fails with No Raw Output:
      * Cause: A critical crash in `pdf_processor.py` or its dependencies happens very early, before output is captured.
      * Fix: Temporarily debug by running `pdf_processor.py` manually inside a container. Mount the input collection (`docker run -it --rm -v "$(pwd)/MyCollection:/app/current_collection" your_image bash`) and then execute `python pdf_processor.py /app/current_collection`. The error message will appear directly in your terminal.

## Team Collaboration

This project is a collaborative effort. `Challenge_1B` integrates and leverages the robust PDF processing and heading classification capabilities developed in `Challenge_1A`. Contributions from both parts are essential for the system's overall functionality and performance.

```
```