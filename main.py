import pdfplumber
import json
import os
import re
import time

# Define the input and output directories as per challenge requirements
INPUT_DIR = '/app/input'
OUTPUT_DIR = '/app/output'

# Heuristic thresholds for heading detection
# These values might need fine-tuning based on your sample PDFs.
# They are relative to the largest font size found on a page or document.
FONT_SIZE_TITLE_THRESHOLD = 1.8 # e.g., 1.8x larger than average text on page 1
FONT_SIZE_H1_THRESHOLD = 1.4    # e.g., 1.4x larger than average text
FONT_SIZE_H2_THRESHOLD = 1.2    # e.g., 1.2x larger than average text
FONT_SIZE_H3_THRESHOLD = 1.05   # e.g., 1.05x larger than average text, or just bold

# Minimum vertical space above a heading to distinguish it from body text
MIN_VERTICAL_SPACE_RATIO = 1.5 # e.g., 1.5x the line height of normal text

# Function to check if a text block is likely bold based on font name
def is_bold(font_name):
    if not font_name:
        return False
    # Common indicators for bold fonts
    return any(keyword in font_name.lower() for keyword in ['bold', 'heavy', 'black', 'demi'])

# Function to calculate average font size and line height for a page
def get_page_text_metrics(page):
    # Filter out very small characters (e.g., superscripts, noise)
    texts = [char for char in page.chars if char['size'] > 5] # Filter out tiny noise
    if not texts:
        return 0, 0, 0

    font_sizes = [char['size'] for char in texts]
    avg_font_size = sum(font_sizes) / len(font_sizes)

    # Estimate average line height from 'doctop' differences of consecutive lines
    line_tops = sorted(list(set([char['doctop'] for char in texts])))
    if len(line_tops) < 2:
        return avg_font_size, avg_font_size, 0 # Fallback if not enough lines

    line_heights = [line_tops[i+1] - line_tops[i] for i in range(len(line_tops) - 1) if (line_tops[i+1] - line_tops[i]) > 0]
    avg_line_height = sum(line_heights) / len(line_heights) if line_heights else avg_font_size
    
    # Get the max font size on the page
    max_font_size_on_page = max(font_sizes)

    return avg_font_size, avg_line_height, max_font_size_on_page

def extract_outline_from_pdf(pdf_path):
    title = "Untitled Document"
    outline = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # --- 1. Extract Title (Heuristic for first page) ---
            if pdf.pages:
                first_page = pdf.pages[0]
                page_text = first_page.extract_text()
                
                # Try to find the largest text block on the first page as title
                # Filter out very small characters and get unique text lines
                chars_on_first_page = [char for char in first_page.chars if char['size'] > 8]
                if chars_on_first_page:
                    # Group characters into text lines
                    text_lines = {}
                    for char in chars_on_first_page:
                        y_pos = round(char['top'], 0) # Group by rounded y-position
                        if y_pos not in text_lines:
                            text_lines[y_pos] = []
                        text_lines[y_pos].append(char)
                    
                    # Sort lines by y-position
                    sorted_y_pos = sorted(text_lines.keys())

                    potential_titles = []
                    for y_pos in sorted_y_pos:
                        line_chars = sorted(text_lines[y_pos], key=lambda x: x['x0'])
                        line_text = "".join([char['text'] for char in line_chars]).strip()
                        if not line_text:
                            continue
                        
                        # Calculate average font size for the line
                        line_font_sizes = [char['size'] for char in line_chars]
                        avg_line_font_size = sum(line_font_sizes) / len(line_font_sizes)
                        
                        potential_titles.append({
                            'text': line_text,
                            'font_size': avg_line_font_size,
                            'y0': line_chars[0]['top'] if line_chars else 0
                        })
                    
                    # Sort potential titles by font size (descending) and y-position (ascending)
                    potential_titles.sort(key=lambda x: (-x['font_size'], x['y0']))

                    if potential_titles:
                        # The very first line, or the largest font on the first page, is a good candidate
                        # We'll take the largest one that appears near the top
                        for pt in potential_titles:
                            # Simple heuristic: if it's significantly larger than others and near the top
                            if pt['font_size'] > (potential_titles[1]['font_size'] * FONT_SIZE_TITLE_THRESHOLD if len(potential_titles) > 1 else 0) and pt['y0'] < first_page.height / 3:
                                title = pt['text']
                                break
                        if title == "Untitled Document" and potential_titles: # Fallback to largest if no clear winner
                             title = potential_titles[0]['text']

            # --- 2. Extract Headings (H1, H2, H3) ---
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # Extract text blocks with layout information
                # Use extract_words or extract_text(x_tolerance=...) for better block detection
                # Here, we'll try to get blocks of text that are likely lines/paragraphs
                
                # Get all text elements (chars) and group them into lines
                page_chars = [char for char in page.chars if char['size'] > 5] # Filter noise
                if not page_chars:
                    continue

                # Group chars into lines based on y-position
                lines_by_y = {}
                for char in page_chars:
                    y_key = round(char['top'], 2) # Use rounded top for grouping
                    if y_key not in lines_by_y:
                        lines_by_y[y_key] = []
                    lines_by_y[y_key].append(char)
                
                sorted_line_y_keys = sorted(lines_by_y.keys())

                # Get page metrics for relative font size comparison
                avg_page_font_size, avg_line_height, max_font_size_on_page = get_page_text_metrics(page)
                if avg_page_font_size == 0: # Skip if no meaningful text
                    continue

                previous_line_bottom = 0 # To calculate vertical spacing

                for y_key in sorted_line_y_keys:
                    line_chars = sorted(lines_by_y[y_key], key=lambda x: x['x0'])
                    line_text = "".join([char['text'] for char in line_chars]).strip()
                    
                    if not line_text:
                        continue

                    # Calculate line's average font size and check for bolding
                    line_font_sizes = [char['size'] for char in line_chars]
                    line_font_names = [char['fontname'] for char in line_chars]
                    
                    avg_line_font_size = sum(line_font_sizes) / len(line_font_sizes) if line_font_sizes else 0
                    is_line_bold = any(is_bold(fn) for fn in line_font_names)

                    # Calculate vertical space above the current line
                    current_line_top = line_chars[0]['top'] if line_chars else y_key
                    vertical_space = current_line_top - previous_line_bottom
                    
                    # Update previous_line_bottom for the next iteration
                    previous_line_bottom = line_chars[-1]['bottom'] if line_chars else y_key + avg_line_height # Estimate bottom

                    # Heuristic for heading identification
                    # A line is a candidate if it's significantly larger or bold
                    is_candidate = False
                    if avg_line_font_size > avg_page_font_size * FONT_SIZE_H3_THRESHOLD or is_line_bold:
                        is_candidate = True
                    
                    # Add a check for significant vertical spacing above the line
                    if avg_line_height > 0 and vertical_space > (avg_line_height * MIN_VERTICAL_SPACE_RATIO):
                        is_candidate = True # More confident it's a heading if there's space

                    if is_candidate:
                        level = None
                        if avg_line_font_size >= max_font_size_on_page * 0.95 and page_num > 1: # Very large, likely H1 (not title)
                            level = "H1"
                        elif avg_line_font_size >= avg_page_font_size * FONT_SIZE_H1_THRESHOLD:
                            level = "H1"
                        elif avg_line_font_size >= avg_page_font_size * FONT_SIZE_H2_THRESHOLD:
                            level = "H2"
                        elif avg_line_font_size >= avg_page_font_size * FONT_SIZE_H3_THRESHOLD and is_line_bold:
                            level = "H3"
                        elif is_line_bold and avg_line_font_size >= avg_page_font_size * 0.9: # Bold and close to avg size
                             level = "H3"

                        if level:
                            # Basic filtering for non-heading lines that might sneak in
                            # e.g., page numbers, running headers/footers
                            if len(line_text) < 5 and re.match(r'^\d+$', line_text): # Likely page number
                                continue
                            if len(line_text) < 10 and (re.match(r'^[A-Z]\.$', line_text) or re.match(r'^\d+\.\d+$', line_text)): # Section numbers like 'A.' or '1.1'
                                continue
                            
                            # Avoid adding duplicates if a line is split by pdfplumber but conceptually one
                            if outline and outline[-1]['text'] == line_text and outline[-1]['page'] == page_num:
                                continue # Skip exact duplicate on same page

                            outline.append({
                                "level": level,
                                "text": line_text,
                                "page": page_num
                            })
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        # Return empty outline if there's an error
        return {"title": "Error Processing Document", "outline": []}

    return {"title": title, "outline": outline}

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process all PDF files in the input directory
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            print(f"Processing '{filename}'...")
            start_time = time.time()
            
            extracted_data = extract_outline_from_pdf(pdf_path)
            
            end_time = time.time()
            print(f"Finished processing '{filename}' in {end_time - start_time:.2f} seconds.")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, ensure_ascii=False, indent=2)
            print(f"Output saved to '{output_path}'")

if __name__ == "__main__":
    main()
