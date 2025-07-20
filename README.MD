PDF Outline Extractor - Connecting the Dots Challenge (Round 1A)
This project provides a solution for Round 1A of the "Connecting the Dots" Challenge, focusing on extracting a structured outline (Title, H1, H2, H3 headings with page numbers) from PDF documents and outputting them in a specified JSON format.

Approach
Our solution employs a heuristic-based approach to identify headings within PDF documents. It leverages pdfplumber to extract text content along with crucial metadata like font size, font name (to infer bolding), and spatial coordinates for each text character.

The core logic involves:

Title Extraction: The document title is identified primarily from the first page, looking for the largest and most prominent text block near the top.

Text Block Analysis: For each page, the script extracts individual text lines and their associated font properties.

Heuristic Classification: Each text line is evaluated against a set of heuristics to determine if it's a potential heading and, if so, its level (H1, H2, or H3). These heuristics consider:

Font Size: Headings are generally larger than body text. Relative font size comparisons are made against the average font size on the page and the maximum font size on the page.

Font Weight (Bolding): The presence of keywords like "Bold", "Heavy", etc., in the font name is used to detect bold text, which is a strong indicator for headings, especially H3.

Vertical Spacing: Headings often have more vertical space above them compared to regular body text, helping to distinguish them.

Positional Information: While not explicitly used for hierarchy, the relative position helps in grouping text.

JSON Output: The extracted title and hierarchical headings (level, text, page number) are then formatted into the required JSON structure.

This heuristic approach is chosen to meet the strict performance (≤ 10 seconds for 50 pages) and model size (≤ 200MB) constraints, as it avoids the overhead of large machine learning models and complex training data.

Libraries Used
pdfplumber: A Python library for extracting text and layout information from PDFs. It provides access to characters, words, lines, and their properties (e.g., font size, bounding boxes), which is essential for our heuristic approach.

json: Python's built-in library for working with JSON data.

os: Python's built-in library for interacting with the operating system (e.g., listing files, creating directories).

re: Python's built-in regular expression module for basic text pattern matching.

time: Python's built-in module for timing execution.

How to Build and Run Your Solution (for documentation purposes)
Prerequisites:

Docker installed on your system.

1. Clone the repository (or create the files manually):

# Assuming you have a Git repository
git clone <your-repo-url>
cd pdf_outline_extractor

(If creating manually, ensure you have the Dockerfile, requirements.txt, main.py, README.md, input/, and output/ directories in your project root.)

2. Place your PDF files:

Put the PDF files you want to process into the input/ directory. For example, input/sample.pdf.

3. Build the Docker Image:

Navigate to the root directory of your project (where Dockerfile is located) and run the following command:

docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
# Replace 'mysolutionname:somerandomidentifier' with your chosen image name and tag.
# Example: docker build --platform linux/amd64 -t pdfextractor:v1.0 .

4. Run the Docker Container:

After building the image, execute your solution using the following command. This command mounts your local input/ and output/ directories to the container's /app/input and /app/output paths, respectively, and disables network access.

docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
# Use the same image name and tag you used during the build step.
# Example: docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdfextractor:v1.0

5. Check the Output:

After the container finishes execution, your generated JSON files will be available in your local output/ directory, corresponding to each PDF file processed. For input/sample.pdf, you should find output/sample.json.