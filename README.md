# UniGuide - Arabic Bylaws NLP Project

## Phase 1: Arabic Data Pipeline

This phase focuses on extracting and structuring Arabic text from the university's PDF regulations.

### Accomplishments:

1. **Extraction (OCR):** Used `EasyOCR` to handle complex Arabic PDF encoding and extract raw text.
2. **Preprocessing:** Applied Arabic normalization using `PyArabic` (removing diacritics, unifying Hamzas and Ta-Marbuta).
3. **Structural Mapping:** Segmented the text into logical JSON chunks based on articles and sections.
4. **Data Validation:** Verified the integrity of the processed text.
5. **Baseline Keyword Matching:** Added Arabic keyword matching with light stemming so forms like `الطالب` and `للطلاب` match the same concept.

### How to run:

1. Install requirements: `pip install -r requirements.txt`
2. Run extraction: `python extract.py`
3. Run preprocessing: `python preprocess.py`
4. Run baseline search: `python arabic_keyword_matcher.py "حقوق الطالب" --top-k 5`

### Baseline search details:

- Normalizes Arabic letters, diacritics, tatweel, Hamza forms, `ى/ي`, and `ة/ه`.
- Removes common Arabic prefixes and suffixes such as `ال`, `لل`, `و`, `ب`, `ات`, `ون`, and pronoun endings.
- Uses keyword overlap scoring over the preprocessed chunks and returns the highest-scoring matches.
