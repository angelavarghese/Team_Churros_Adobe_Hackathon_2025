import os
import json
import subprocess
import zipfile
import shutil
import tempfile
from flask import Flask, request, render_template, send_from_directory, flash, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here_for_flash_messages' 
app.config['UPLOAD_FOLDER'] = '/tmp/uploads' # Temporary upload folder inside container
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # Max upload size

# Only one allowed extension for the main upload (zip)
ALLOWED_FILE_EXTENSIONS = {'zip'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define expected input formats for display on the page
EXPECTED_DIR_FORMAT = """
Your uploaded ZIP file should contain:
- A single top-level folder (e.g., 'MyCollection')
  - Inside this top-level folder:
    - 'PDFs/' folder (containing all your PDF files)
    - 'input.json' file
"""

EXPECTED_INPUT_JSON_FORMAT = """
{
    "documents": [
        {
            "filename": "Bhagavad_Gita_Japanese.pdf",
            "title": "Bhagavad Gita (Japanese Translation)"
        },
        {
            "filename": "Another_Doc.pdf",
            "title": "Another Document Title"
        }
    ],
    "persona": {
        "role": "Comparative Religious Scholar"
    },
    "job_to_be_done": {
        "task": "Identify and summarize key philosophical concepts related to Dharma, Karma, and self-realization from the text."
    }
}
"""

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload form page."""
    return render_template(
        'index.html',
        dir_format=EXPECTED_DIR_FORMAT,
        json_format=EXPECTED_INPUT_JSON_FORMAT,
        results=None
    )

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE_EXTENSIONS

@app.route('/process', methods=['POST'])
def process_upload():
    """Handles the uploaded ZIP file and processes it."""
    if 'file' not in request.files: # Changed from 'pdfs_zip' or 'input_json_file'
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file'] # Changed from pdfs_zip_file or input_json_file
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)

        # Create a temporary directory to extract the zip contents
        temp_extraction_base_dir = tempfile.mkdtemp() # Base dir for extraction
        actual_collection_path = None # Initialize for finally block

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extraction_base_dir)

            # Find the actual collection folder inside the zip (assumed to be the only top-level dir)
            extracted_contents = os.listdir(temp_extraction_base_dir)
            if len(extracted_contents) != 1 or not os.path.isdir(os.path.join(temp_extraction_base_dir, extracted_contents[0])):
                flash('ZIP file must contain exactly one top-level collection folder (e.g., MyCollection/).')
                return render_template(
                    'index.html',
                    dir_format=EXPECTED_DIR_FORMAT,
                    json_format=EXPECTED_INPUT_JSON_FORMAT,
                    results=None,
                    error_message="Invalid ZIP structure. Please ensure it contains a single top-level collection folder."
                )
            
            collection_folder_name = extracted_contents[0]
            actual_collection_path = os.path.join(temp_extraction_base_dir, collection_folder_name)

            # Validate expected files within the collection folder
            pdfs_dir = os.path.join(actual_collection_path, 'PDFs')
            input_json_file = os.path.join(actual_collection_path, 'input.json')

            if not os.path.exists(pdfs_dir) or not os.path.isdir(pdfs_dir):
                flash('Extracted collection folder missing "PDFs/" directory.')
                return render_template(
                    'index.html',
                    dir_format=EXPECTED_DIR_FORMAT,
                    json_format=EXPECTED_INPUT_JSON_FORMAT,
                    results=None,
                    error_message="Extracted collection folder missing 'PDFs/' directory."
                )
            if not os.path.exists(input_json_file) or not os.path.isfile(input_json_file):
                flash('Extracted collection folder missing "input.json" file.')
                return render_template(
                    'index.html',
                    dir_format=EXPECTED_DIR_FORMAT,
                    json_format=EXPECTED_INPUT_JSON_FORMAT,
                    results=None,
                    error_message="Extracted collection folder missing 'input.json' file."
                )

            # --- Call pdf_processor.py using subprocess ---
            pdf_processor_script_path = os.path.join(app.root_path, 'pdf_processor.py')
            
            # Pass the absolute path of the *extracted collection folder* to pdf_processor.py
            cmd = ['python', pdf_processor_script_path, actual_collection_path]
            
            # Execute the script and capture its output
            process = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to capture stderr
            
            raw_cli_output = process.stdout + process.stderr
            if process.returncode != 0:
                flash(f"Processing failed. Check raw output below for details.")
                return render_template(
                    'index.html',
                    dir_format=EXPECTED_DIR_FORMAT,
                    json_format=EXPECTED_INPUT_JSON_FORMAT,
                    results=None,
                    error_message=f"Document processing failed (exit code {process.returncode}). Check raw output below.",
                    raw_cli_output=raw_cli_output
                )

            # Read the output.json generated by pdf_processor.py (located within the actual_collection_path)
            output_json_path = os.path.join(actual_collection_path, 'output.json')
            if not os.path.exists(output_json_path):
                flash('Processing completed, but output.json was not generated.')
                return render_template(
                    'index.html',
                    dir_format=EXPECTED_DIR_FORMAT,
                    json_format=EXPECTED_INPUT_JSON_FORMAT,
                    results=None,
                    error_message="Processing completed, but 'output.json' was not generated. Check raw output below."
                )

            with open(output_json_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)

            # --- Prepare Data for Display ---
            num_pdfs = len(output_data.get('metadata', {}).get('input_documents', []))
            num_extracted_sections = len(output_data.get('extracted_sections', []))
            
            subsection_analysis_paragraphs = []
            for item in output_data.get('subsection_analysis', []):
                subsection_analysis_paragraphs.append(
                    f"<strong>Document:</strong> {item.get('document', 'N/A')}<br>"
                    f"<strong>Page:</strong> {item.get('page_number', 'N/A')}<br>"
                    f"<strong>Refined Text:</strong> {item.get('refined_text', 'N/A')}"
                )
            
            # For download button: store output data in a global variable (simple for demo)
            global LAST_OUTPUT_JSON_DATA 
            LAST_OUTPUT_JSON_DATA = output_data

            flash('Processing completed successfully!')
            return render_template(
                'index.html',
                dir_format=EXPECTED_DIR_FORMAT,
                json_format=EXPECTED_INPUT_JSON_FORMAT,
                results={
                    'output_json_data': json.dumps(output_data, indent=4, ensure_ascii=False),
                    'subsection_analysis_paragraphs': subsection_analysis_paragraphs,
                    'num_pdfs': num_pdfs,
                    'num_extracted_sections': num_extracted_sections,
                    'raw_cli_output': raw_cli_output
                }
            )

        except zipfile.BadZipFile:
            flash('Invalid ZIP file format. Please upload a valid .zip file.')
            return render_template(
                'index.html',
                dir_format=EXPECTED_DIR_FORMAT,
                json_format=EXPECTED_INPUT_JSON_FORMAT,
                results=None,
                error_message="Invalid ZIP file format. Please upload a valid .zip file."
            )
        except Exception as e:
            flash(f'An unexpected error occurred: {e}')
            current_raw_cli_output = raw_cli_output if 'raw_cli_output' in locals() else "No CLI output captured due to early error."
            return render_template(
                'index.html',
                dir_format=EXPECTED_DIR_FORMAT,
                json_format=EXPECTED_INPUT_JSON_FORMAT,
                results=None,
                error_message=f"An unexpected error occurred during processing: {e}",
                raw_cli_output=current_raw_cli_output
            )
        finally:
            # Clean up the temporary directory and uploaded zip file
            if 'temp_extraction_base_dir' in locals() and os.path.exists(temp_extraction_base_dir):
                shutil.rmtree(temp_extraction_base_dir)
            if 'zip_path' in locals() and os.path.exists(zip_path):
                os.remove(zip_path)


LAST_OUTPUT_JSON_DATA = {}
@app.route('/download_output_json')
def download_output_json():
    if LAST_OUTPUT_JSON_DATA:
        temp_dir = tempfile.mkdtemp()
        temp_output_file = os.path.join(temp_dir, 'output.json')
        with open(temp_output_file, 'w', encoding='utf-8') as f:
            json.dump(LAST_OUTPUT_JSON_DATA, f, indent=4, ensure_ascii=False)
        
        response = send_from_directory(temp_dir, 'output.json', as_attachment=True)
        @response.call_on_close
        def cleanup():
            shutil.rmtree(temp_dir)
        return response
    else:
        flash('No output.json available for download. Please process a collection first.')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)