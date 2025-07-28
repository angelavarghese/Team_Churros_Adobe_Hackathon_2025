# Team Churros - Adobe Hackathon 2025

## Project Structure

The repository is organized into two main challenge folders:

```

Team\_Churros\_Adobe\_Hackathon\_2025/
├── Challenge\_1A/                               \# Teammate's PDF Processing Application
│   ├── robust\_pdf\_extractor.py                 \# Core PDF analysis (detect type, OCR, extract features)
│   ├── predict\_and\_export.py                   \# ML inference for heading classification
│   ├── extract\_features.py                     \# Feature extraction logic for ML model
│   ├── model/                                  \# Trained ML models and label maps
│   │   ├── model.joblib
│   │   └── label\_map.json
│   └── **init**.py                             \# Makes Challenge\_1A a Python package
│
└── Challenge\_1B/                               \# Your Persona-Driven Document Intelligence App
├── Collection 1/                           \# Example input collection (for CLI or testing)
│   ├── PDFs/                               \#   -\> Contains PDF documents
│   │   └── doc1.pdf
│   └── input.json                          \#   -\> Defines documents, persona, job-to-be-done
└── Collection 2/                           \# Another example collection
├── PDFs/
└── input.json
├── Dockerfile                              \# Dockerfile for CLI version (pdf\_processor.py only)
├── Dockerfile\_flask                        \# Dockerfile for Flask web application
├── README.md                               \# This file
├── app.py                                  \# Flask web application entry point
├── flaskapp.py                             \# Alias for app.py (from your provided list)
├── pdf\_processor.py                        \# Core document intelligence logic
├── requirements.txt                        \# Python dependencies
├── templates/                              \# HTML templates for Flask app
│   └── index.html
└── .dockerignore                           \# Files/folders to ignore in Docker builds & Git
└── huggingface\_cache/                      \# Local cache for SentenceTransformer models (ignored by Git/Docker build)

````

## Prerequisites

Before you begin, ensure you have the following installed:

* **Git:** For cloning the repository.
* **Docker Desktop:** (For Windows/macOS) Provides the Docker engine and WSL2 integration.
* **Docker Engine:** (For Linux)
* **Python 3.12 or higher:** It's recommended to use a virtual environment.

### Docker Desktop Resources (Crucial for Building)

The ML models (SentenceTransformers, RandomForest) and OCR tools (Tesseract, Poppler) are memory-intensive during installation. To avoid `SIGBUS` errors during Docker builds, **ensure Docker Desktop is allocated sufficient resources**:

1.  Open Docker Desktop Settings.
2.  Go to **Resources** (or **Advanced**).
3.  Set **Memory** to at least **6GB - 8GB** (or more, if your system has spare RAM).
4.  Set **CPUs** to at least **4**.
5.  Click **Apply & Restart**.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your_github_username/Team_Churros_Adobe_Hackathon_2025.git](https://github.com/your_github_username/Team_Churros_Adobe_Hackathon_2025.git)
    cd Team_Churros_Adobe_Hackathon_2025/Challenge_1B/
    ```

2.  **Create Python Virtual Environment (Local Development):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows CMD: .\venv\Scripts\activate
    ```

3.  **Install Python Dependencies (Local Development - for basic local testing if needed):**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This `venv` is for local development/testing. Docker builds its own environment.*
