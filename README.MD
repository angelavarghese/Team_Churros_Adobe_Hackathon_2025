# PDF Outline Extractor

## Overview
This project extracts a structured outline (Title, H1, H2, H3 headings with page numbers) from PDF documents and outputs them in a specific JSON format. It is designed for the "Connecting the Dots" Challenge (Round 1A) and runs as a Dockerized Python application.

## What Has Been Implemented
- **Heuristic-based heading detection** using font size, boldness, vertical spacing, and position (via `pdfplumber`).
- **Title extraction** from the first page (largest, topmost text).
- **Processes all PDFs** in `/app/input` and writes JSON outlines to `/app/output`.
- **Offline, CPU-only, and lightweight** (≤200MB dependencies, no internet required).
- **Performance-optimized** to process a 50-page PDF in ≤10 seconds.

## How It Works
1. **Title Extraction:** Finds the largest, most prominent text block near the top of the first page.
2. **Heading Detection:** For each page, analyzes each text line's font size, boldness, and vertical spacing to classify as H1, H2, or H3.
3. **Filtering:** Skips page numbers, section numbers, and duplicate headings.
4. **Output:** Writes a JSON file for each PDF, containing the title and a list of headings with their levels and page numbers.

## How to Run (Step-by-Step)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

### 1. Clone the Repository
```powershell
git clone https://github.com/Raged-Pineapple/adobe-repo.git
cd adobe-repo
```

### 2. Prepare Input/Output Folders
```powershell
New-Item -ItemType Directory -Path "input" -Force
New-Item -ItemType Directory -Path "output" -Force
```

### 3. Add PDF Files
Place your PDF files in the `input` folder (e.g., `input/sample.pdf`).

### 4. Build the Docker Image
```powershell
docker build --platform linux/amd64 -t pdfextractor:v1.0 .
```

### 5. Run the Application
```powershell
docker run --rm -v ${PWD}\input:/app/input -v ${PWD}\output:/app/output --network none pdfextractor:v1.0
```
If you get an error with `${PWD}`, use the full path:
```powershell
docker run --rm -v D:\adobehack\input:/app/input -v D:\adobehack\output:/app/output --network none pdfextractor:v1.0
```

### 6. Check the Output
- Each PDF in `input` will have a corresponding `.json` file in `output`.
- Example: `input/sample.pdf` → `output/sample.json`

## Output Format Example
```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Section 1", "page": 1},
    {"level": "H2", "text": "Subsection", "page": 2}
  ]
}
```

## Notes
- No internet access is required during execution.
- Only CPU resources are used (no GPU needed).
- For best results, fine-tune the heuristics in `main.py` if your PDFs have unusual formatting.