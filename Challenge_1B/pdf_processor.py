import fitz
import os
import sys
import regex # Using the 'regex' module directly for full Unicode support
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from datetime import datetime
import warnings # For suppressing future warnings

# Suppress all warnings (useful for cleaner CLI output in production)
warnings.filterwarnings("ignore")

# NEW: Import Teammate's Module from the 'Challenge_1A' package
# Assuming robust_pdf_extractor.py and predict_and_export.py are directly under Challenge_1A/
# The 'Challenge_1A' package is copied to /app/Challenge_1A/ in the Docker image.
from Challenge_1A import robust_pdf_extractor

# --- Configuration Constants ---
HF_MODELS_CACHE_DIR = "/app/huggingface_cache" 
OUTPUT_FILENAME = "output.json"

# --- Compiled Regex Pattern (for clean_text_for_output) ---
_unicode_allowed_chars_pattern = regex.compile(
    r'[^\p{L}\p{N}\s.,!?;:\'"()\-\[\]{}&%$#@*+/=<>|`~]', 
    flags=regex.UNICODE
)

# --- Core PDF Processing Functions ---
def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF document, page by page.
    This function is primarily for fallback if teammate's app fails.
    """
    pages_text = []
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pages_text.append(page.get_text("text"))
        doc.close()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return []
    return pages_text


def get_document_paths_from_input(input_documents_list, base_dir):
    """
    Constructs full PDF paths based on the input JSON documents list and a base directory.
    Checks if each file actually exists.
    """
    pdf_files = []
    for doc_info in input_documents_list:
        filename = doc_info.get("filename")
        if filename:
            full_path = os.path.join(base_dir, filename)
            if os.path.isfile(full_path):
                pdf_files.append(full_path)
            else:
                print(f"Warning: PDF file not found at expected path: {full_path}")
        else:
            print(f"Warning: Document entry missing 'filename': {doc_info}")
    return pdf_files

def get_exclusion_keywords(persona_description, job_to_be_done):
    """
    Returns a list of keywords that should be excluded from refined text,
    based on the persona and job.
    """
    exclusions = []
    
    # --- Vegetarian/Veggie Exclusion ---
    if regex.search(r'\b(vegetarian|veggie|plant-based)\b', persona_description.lower()) or \
       regex.search(r'\b(vegetarian|veggie|plant-based)\b', job_to_be_done.lower()):
        exclusions.extend([
            "meat", "beef", "pork", "chicken", "lamb", "fish", "bacon", "sausage", "ham",
            "turkey", "seafood", "salmon", "tuna", "shrimp", "oyster", "gelatin", "lard",
            "animal fat", "flesh", "poultry", "venison", "shellfish", "crab", "lobster",
            "calamari", "squid", "mussels", "clams", "prawns", "anchovies", "halibut",
            "cod", "trout", "duck", "goose"
        ])
    
    # --- Gluten-Free Exclusion ---
    if regex.search(r'\b(gluten-free|celiac)\b', persona_description.lower()) or \
       regex.search(r'\b(gluten-free|celiac)\b', job_to_be_done.lower()):
        exclusions.extend([
            "wheat", "rye", "barley", "oats", "malt", "brewer's yeast", "triticale",
            "semolina", "durum", "spelt", "farro", "kamut", "couscous", "bulgur",
            "bread", "pasta", "flour", "gluten"
        ])
    
    return [kw.lower() for kw in exclusions]

def clean_text_for_output(text):
    """
    Removes newline characters and limits characters to Unicode alphabets, numbers, and common punctuation.
    """
    # 1. Replace all whitespace (including newlines) with a single space
    cleaned_text = regex.sub(r'\s+', ' ', text) # Use regex.sub
    
    # 2. Keep only alphanumeric characters (Unicode) and common punctuation.
    # Use the pre-compiled pattern's sub method.
    cleaned_text = _unicode_allowed_chars_pattern.sub('', cleaned_text)
    
    # 3. Remove multiple spaces that might result from cleaning
    cleaned_text = regex.sub(r' +', ' ', cleaned_text).strip() # Use regex.sub

    return cleaned_text


# --- NEW FUNCTION: Integrate with Teammate's PDF Extractor ---
def get_sections_from_teammate_app(pdf_path, document_name):
    """
    Calls the robust_pdf_extractor to get structured outline and text blocks.
    Parses its output into our system's expected section format.
    """
    extracted_sections_from_teammate = []
    teammate_app_outline = [] # To store the outline from teammate's app for printing
    try:
        # Call the main processing function from your teammate's module
        # This function processes the PDF, extracts features, classifies, and returns an outline.
        teammate_output = robust_pdf_extractor.process_pdf(pdf_path)

        if teammate_output and "outline" in teammate_output:
            teammate_app_outline = teammate_output["outline"] # Store the outline

            # Re-extract all blocks from PyMuPDF again to get full text content
            doc = fitz.open(pdf_path)
            all_blocks_with_text = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                blocks_on_page = page.get_text("blocks") # (x0, y0, x1, y1, text, block_no, block_type)
                for block in blocks_on_page:
                    # Use robust_pdf_extractor's normalize_text for consistency
                    block_text = robust_pdf_extractor.normalize_text(block[4])
                    if block_text.strip(): # Only consider non-empty blocks
                        all_blocks_with_text.append({
                            "page_num": page_num + 1,
                            "text": block_text,
                            "bbox": block[:4] # Store bounding box for positional awareness
                        })
            doc.close()

            # Now, iterate through the blocks and group them under the headings identified by the teammate's app
            current_heading_text = None
            current_heading_page = -1
            current_section_content_blocks = []

            # Sort blocks by page and then y-position (reading order) for reliable content grouping
            all_blocks_with_text.sort(key=lambda b: (b['page_num'], b['bbox'][1]))

            # Create a set of identified heading texts for quick lookup
            # Use robust_pdf_extractor's normalize_text for robust matching
            identified_headings_set = set(robust_pdf_extractor.normalize_text(h['text']) for h in teammate_output['outline'])

            for block in all_blocks_with_text:
                block_page = block['page_num']
                block_text = block['text']
                normalized_block_text = robust_pdf_extractor.normalize_text(block_text)

                is_new_heading = False
                if normalized_block_text in identified_headings_set:
                     # More precise check: Ensure it matches an actual heading from the outline
                    for heading_item in teammate_output["outline"]:
                        if heading_item["page"] == block_page and robust_pdf_extractor.normalize_text(heading_item["text"]) == normalized_block_text:
                            is_new_heading = True
                            break

                if is_new_heading:
                    # If a new heading is found and we have accumulated content for the *previous* section, save it
                    if current_heading_text and current_section_content_blocks:
                        extracted_sections_from_teammate.append({
                            "document": document_name,
                            "section_title": current_heading_text,
                            "importance_rank": -1,
                            "page_number": current_heading_page,
                            "text_content": "\n".join(current_section_content_blocks).strip()
                        })
                    # Start a new section with the current heading
                    current_heading_text = block_text # Use original block_text as heading title
                    current_heading_page = block_page
                    current_section_content_blocks = [] # Reset content blocks for the new section
                else:
                    # If it's not a heading, add it to the current section's content
                    current_section_content_blocks.append(block_text)

            # Add the last accumulated section after the loop finishes
            if current_heading_text and current_section_content_blocks:
                extracted_sections_from_teammate.append({
                    "document": document_name,
                    "section_title": current_heading_text,
                    "importance_rank": -1,
                    "page_number": current_heading_page,
                    "text_content": "\n".join(current_section_content_blocks).strip()
                })
            # Fallback if no headings were found by teammate's app or if it's just one large block of text
            elif not extracted_sections_from_teammate and all_blocks_with_text:
                 extracted_sections_from_teammate.append({
                    "document": document_name,
                    "section_title": f"Document Text (Page {all_blocks_with_text[0]['page_num']})",
                    "importance_rank": -1,
                    "page_number": all_blocks_with_text[0]['page_num'],
                    "text_content": "\n".join([b['text'] for b in all_blocks_with_text]).strip()
                })


            print(f"Successfully extracted {len(extracted_sections_from_teammate)} sections via teammate's app outline.")
        else:
            print(f"Teammate's app returned no outline for '{document_name}'.")
            # Raising an error here to trigger the outer except block for better fallback handling.
            raise ValueError("No outline provided by teammate's app, triggering fallback.")

    except Exception as e:
        print(f"Error integrating with teammate's app for '{pdf_path}': {e}")
        print("Falling back to basic text extraction (each page becomes a section).")
        pages_content = extract_text_from_pdf(pdf_path) # Our original basic page text extractor
        if pages_content:
            for page_num, text_on_page in enumerate(pages_content):
                if text_on_page.strip():
                    extracted_sections_from_teammate.append({
                        "document": document_name,
                        "section_title": f"Page {page_num + 1} (Fallback)",
                        "importance_rank": -1,
                        "page_number": page_num + 1,
                        "text_content": text_on_page
                    })
            print(f"  (Fallback) Identified {len(extracted_sections_from_teammate)} sections from '{document_name}'.")
    
    return extracted_sections_from_teammate, teammate_app_outline


# --- Main Execution Block ---

if __name__ == "__main__":
    # The script now expects the absolute path to the collection folder as an argument
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <absolute_path_to_collection_folder>")
        print("Example: python pdf_processor.py /tmp/my_collection")
        sys.exit(1)

    # The argument is the absolute path to the collection folder (e.g., /tmp/tmpXYZ/MyCollection)
    collection_folder_path = sys.argv[1]
    collection_name = os.path.basename(collection_folder_path) # Derive name for logging

    print(f"Processing data for collection: {collection_name} at path: {collection_folder_path}")

    # Dynamically construct paths relative to the passed collection_folder_path
    INPUT_JSON_PATH = os.path.join(collection_folder_path, "input.json")
    PDF_INPUT_BASE_DIR = os.path.join(collection_folder_path, "PDFs")
    output_file_path = os.path.join(collection_folder_path, OUTPUT_FILENAME)
    
    # Ensure the output directory (which is the collection folder itself) exists
    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)


    print(f"Loading input from {INPUT_JSON_PATH}...")
    try:
        with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print("Input JSON loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_JSON_PATH}' not found. Please ensure it exists in the collection folder.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding input JSON: {e}. Please check '{INPUT_JSON_PATH}' for syntax errors.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading input JSON: {e}")
        sys.exit(1)

    input_documents_list = input_data.get("documents", [])
    persona_role = input_data.get("persona", {}).get("role", "Default Persona")
    job_task = input_data.get("job_to_be_done", {}).get("task", "Default Job")

    persona_description = persona_role
    job_to_be_done = job_task

    exclusion_keywords = get_exclusion_keywords(persona_description, job_to_be_done)
    if exclusion_keywords:
        print(f"Exclusion keywords identified based on persona/job: {', '.join(exclusion_keywords)}")


    print(f"Persona: {persona_description}")
    print(f"Job: {job_to_be_done}")

    print("Loading SentenceTransformer model...")
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', cache_folder=HF_MODELS_CACHE_DIR)
        print("SentenceTransformer model loaded successfully.")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        print("Ensure the model was downloaded during Docker build and HF_MODELS_CACHE_DIR is correct.")
        sys.exit(1)

    all_pdf_paths = get_document_paths_from_input(input_documents_list, PDF_INPUT_BASE_DIR)
    original_input_filenames = [doc["filename"] for doc in input_documents_list if "filename" in doc]

    if not all_pdf_paths:
        print(f"No valid PDFs found based on input JSON and base directory: '{PDF_INPUT_BASE_DIR}'. Exiting.")
        sys.exit(1)

    all_extracted_sections = []
    all_teammate_outlines = {} 
    
    for pdf_path in all_pdf_paths:
        document_name = os.path.basename(pdf_path)
        print(f"\n--- Processing '{document_name}' using NEW Teammate App Integration ---")
        
        sections_for_doc, teammate_outline = get_sections_from_teammate_app(pdf_path, document_name)
        
        if sections_for_doc:
            all_extracted_sections.extend(sections_for_doc)
            print(f"  Successfully incorporated {len(sections_for_doc)} sections from '{document_name}'.")
        else:
            print(f"  No sections incorporated for '{document_name}' (teammate's app or fallback failed).")
        
        if teammate_outline:
            all_teammate_outlines[document_name] = teammate_outline


    if not all_extracted_sections:
        print("No sections identified from any document. Please check PDF content and sectioning heuristics. Exiting.")
        sys.exit(1)

    print(f"\nTotal identified sections across all documents: {len(all_extracted_sections)}.")

    # --- Display Teammate's Outlines (Console Output) ---
    if all_teammate_outlines:
        print("\n=== Teammate's App - Headings/Outline Output ===")
        for doc_name, outline in all_teammate_outlines.items():
            print(f"\nOutline for '{doc_name}':")
            if outline:
                for i, item in enumerate(outline):
                    level_info = f"[{item['level']}]" if 'level' in item else ""
                    print(f"  {i+1}. {level_info} {item['text']} (Page {item['page']})")
            else:
                print("  (No headings found by teammate's app for this document.)")
        print("==================================================")


    # --- Generate Embeddings ---
    print("\nGenerating embeddings for persona, job, and document sections...")
    persona_job_query = f"Persona: {persona_description}. Task: {job_to_be_done}"
    query_embedding = model.encode(persona_job_query, convert_to_tensor=False)

    section_texts = [s["text_content"] for s in all_extracted_sections]
    section_embeddings = model.encode(section_texts, convert_to_tensor=False)
    print("Embeddings generated.")

    # --- Relevance Scoring for ALL sections ---
    similarities = cosine_similarity(query_embedding.reshape(1, -1), section_embeddings)[0]

    for i, section in enumerate(all_extracted_sections):
        section["similarity_score"] = similarities[i]


    # --- Select the single most relevant section per PDF ---
    print("Selecting the most relevant section from each PDF for 'extracted_sections' output...")
    selected_sections_for_output = []
    
    sections_by_document = {}
    for section in all_extracted_sections:
        doc_name = section["document"]
        if doc_name not in sections_by_document:
            sections_by_document[doc_name] = []
        sections_by_document[doc_name].append(section)
    
    for doc_name, sections_list in sections_by_document.items():
        if sections_list:
            most_relevant_section_in_doc = sorted(sections_list, key=lambda x: x["similarity_score"], reverse=True)[0]
            selected_sections_for_output.append(most_relevant_section_in_doc)
    
    ranked_sections_for_output = sorted(selected_sections_for_output, key=lambda x: x["similarity_score"], reverse=True)

    for i, section in enumerate(ranked_sections_for_output):
        section["importance_rank"] = i + 1

    print(f"Selected {len(ranked_sections_for_output)} top sections (one per unique document).")

    # --- Subsection Analysis (from EACH of the selected sections, with keyword filtering & cleaning) ---
    subsection_analysis_output = []

    print(f"Performing subsection analysis for all {len(ranked_sections_for_output)} selected sections (applying keyword filters and text cleaning)...")
    for i, section in enumerate(ranked_sections_for_output):
        section_text = section["text_content"]
        
        sentences = [s.strip() for s in section_text.split('.') if s.strip()]
        
        filtered_sentences = []
        for sentence in sentences:
            should_exclude = False
            sentence_lower = sentence.lower()
            for kw in exclusion_keywords:
                if regex.search(r'\b' + regex.escape(kw) + r'\b', sentence_lower):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_sentences.append(sentence)
        
        sentences_to_embed = filtered_sentences 
        
        if not sentences_to_embed:
            print(f"  Skipping subsection analysis for '{section['document']}' (Page {section['page_num']}) due to all sentences being filtered out.")
            continue # Skip to the next section in the loop

        sentence_embeddings = model.encode(sentences_to_embed, convert_to_tensor=False)
        sentence_similarities = cosine_similarity(query_embedding.reshape(1, -1), sentence_embeddings)[0]

        num_sentences_to_extract = min(3, len(sentences_to_embed))
        top_sentence_indices = np.argsort(sentence_similarities)[::-1][:num_sentences_to_extract]
        
        extracted_sentences_in_order = sorted([sentences_to_embed[idx] for idx in top_sentence_indices], 
                                              key=lambda x: sentences_to_embed.index(x))
        
        cleaned_extracted_sentences = [clean_text_for_output(s) for s in extracted_sentences_in_order]
        
        refined_text = ". ".join(cleaned_extracted_sentences)
        if refined_text and not refined_text.endswith('.'):
            refined_text += "."

        if refined_text:
            subsection_analysis_output.append({
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"]
            })
        else:
            print(f"  Skipping subsection analysis for '{section['document']}' (Page {section['page_number']}) because refined_text became empty after cleaning.")

    print("Subsection analysis complete.")

    # --- Format Final Output as JSON and Save to File ---
    final_output = {
        "metadata": {
            "input_documents": original_input_filenames,
            "persona": persona_description,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": subsection_analysis_output
    }

    for section in ranked_sections_for_output:
        final_output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": section["importance_rank"],
            "page_number": section["page_number"]
        })

    os.makedirs(os.path.dirname(output_file_path) or ".", exist_ok=True)
    
    print(f"\nSaving final output to '{output_file_path}'...")
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print("Output saved successfully!")
    except IOError as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving: {e}")
        sys.exit(1)