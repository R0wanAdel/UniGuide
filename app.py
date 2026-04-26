import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display

def fix_arabic_text(text):
    """
    Reshapes disconnected Arabic letters and reverses them to RTL.
    """
    if not text:
        return ""
    # Reshape to connect the letters
    reshaped_text = arabic_reshaper.reshape(str(text))
    # Apply BiDi algorithm to render Right-to-Left
    bidi_text = get_display(reshaped_text)
    return bidi_text

def extract_bylaws_locally(pdf_path):
    extracted_document = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_content = [f"--- Page {page_number} ---"]
            
            # --- 1. HANDLE TABLES ---
            # Extract tables first and format them as Markdown for the LLM
            tables = page.extract_tables()
            for table in tables:
                md_table = []
                for row_idx, row in enumerate(table):
                    # Clean and fix Arabic in each cell
                    cleaned_row = [fix_arabic_text(cell).replace('\n', ' ') if cell else "" for cell in row]
                    row_string = "| " + " | ".join(cleaned_row) + " |"
                    md_table.append(row_string)
                    
                    # Add Markdown header separator after the first row
                    if row_idx == 0:
                        separator = "| " + " | ".join(["---"] * len(cleaned_row)) + " |"
                        md_table.append(separator)
                
                page_content.append("\n".join(md_table) + "\n")
            
            # --- 2. HANDLE STANDARD TEXT ---
            # Extract text (Note: pdfplumber extracts table text here too, 
            # but having the structured Markdown table above helps the LLM)
            raw_text = page.extract_text()
            if raw_text:
                fixed_lines = []
                for line in raw_text.split('\n'):
                    fixed_lines.append(fix_arabic_text(line))
                
                page_content.append("\n".join(fixed_lines))
            
            # Combine page elements
            extracted_document.append("\n".join(page_content))
            
    return "\n\n".join(extracted_document)

if __name__ == "__main__":
    pdf_file = "كلية-الحاسبات-والمعلومات.pdf"
    
    print("Extracting and fixing Arabic text locally...")
    final_text = extract_bylaws_locally(pdf_file)
    
    with open("extracted_bylaws.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print("Extraction complete. Saved to extracted_bylaws.txt")