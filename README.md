# UniGuide - Arabic Bylaws NLP Project

## Phase 1: Arabic Data Pipeline
This phase focuses on extracting and structuring Arabic text from the university's PDF regulations.

### Accomplishments:
1. **Extraction (OCR):** Used `EasyOCR` to handle complex Arabic PDF encoding and extract raw text.
2. **Preprocessing:** Applied Arabic normalization using `PyArabic` (removing diacritics, unifying Hamzas and Ta-Marbuta).
3. **Structural Mapping:** Segmented the text into logical JSON chunks based on articles and sections.
4. **Data Validation:** Verified the integrity of the processed text.

### How to run:
1. Install requirements: `pip install easyocr pyarabic pymupdf`
2. Run extraction: `python extract.py`
3. Run preprocessing: `python preprocess.py`