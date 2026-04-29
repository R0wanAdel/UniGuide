import pdfplumber
import pytesseract
import arabic_reshaper
from bidi.algorithm import get_display


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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
            
            # Try native text extraction first to save time
            native_text = page.extract_text()
            
            # Only run OCR if native extraction fails or returns very little text
            if not native_text or len(native_text.strip()) < 10:
                page_image = page.to_image(resolution=300).original
                raw_text = pytesseract.image_to_string(page_image, lang='ara')
                if raw_text:
                    fixed_lines = [fix_arabic_text(line) for line in raw_text.split('\n') if line.strip()]
                    page_content.append("\n".join(fixed_lines))
            else:
                # Use native text (still needs reshaping for Arabic)
                page_content.append(fix_arabic_text(native_text))

            # ... (table handling logic can stay here)
            
            extracted_document.append("\n".join(page_content))
            
    return "\n\n".join(extracted_document)


if __name__ == "__main__":
    pdf_file = "كلية-الحاسبات-والمعلومات.pdf"
    
    print("Extracting and fixing Arabic text locally using OCR...")
    final_text = extract_bylaws_locally(pdf_file)
    
    with open("extracted_bylaws.txt", "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print("Extraction complete. Saved to extracted_bylaws.txt")