# UniGuide - Arabic Bylaws NLP Project

## Phase 1: Arabic Data Pipeline

This project extracts and structures Arabic university bylaw text, then supports keyword search, semantic search, and answer evaluation.

### Accomplishments

1. **Extraction (OCR):** Uses `EasyOCR` to handle Arabic PDF pages and extract raw text.
2. **Preprocessing:** Applies Arabic normalization with `PyArabic`, including diacritics removal and letter unification.
3. **Structural Mapping:** Segments the bylaws into JSON chunks based on articles and sections.
4. **Baseline Keyword Matching:** Adds Arabic keyword matching with light stemming.
5. **Semantic Search:** Uses multilingual sentence embeddings with Arabic keyword boosts.
6. **Evaluator:** Tracks Accuracy, F1 Score, and Exact Match while accounting for Arabic synonyms.

### How to run

1. Install requirements: `pip install -r requirements.txt`
2. Run extraction: `python extract.py`
3. Run preprocessing: `python preprocess.py`
4. Run baseline search: `python arabic_keyword_matcher.py "حقوق الطالب" --top-k 5`
5. Run semantic search: `python stage7_semantic_search.py "الحد الاقصى للساعات المعتمدة" --top-k 5`
6. Run evaluator: `python stage8_evaluator.py evaluation_samples.json --details`

### Baseline search details

- Normalizes Arabic letters, diacritics, tatweel, Hamza forms, `ى/ي`, and `ة/ه`.
- Removes common Arabic prefixes and suffixes such as `ال`, `لل`, `و`, `ب`, `ات`, `ون`, and pronoun endings.
- Uses keyword overlap scoring over the preprocessed chunks and returns the highest-scoring matches.

### Stage 8 evaluator

- Tracks `Accuracy`, `F1 Score`, and `Exact Match` for predicted answers against reference answers.
- Normalizes Arabic spelling variants, diacritics, tatweel, punctuation, and Arabic-Indic digits before scoring.
- Accounts for Arabic synonyms and variants such as `طالب/طلاب/طلبة`, `ساعة/ساعات`, `مقرر/مادة`, `تسجيل/قيد`, and `أدنى/أقل`.
- Accepts `.json`, `.jsonl`, or `.csv` files with fields such as `prediction` and `references`/`reference`/`ground_truth`.
- If `expected_chunk_id` and `predicted_chunk_id` are present, accuracy uses retrieval correctness; otherwise it uses synonym-aware exact match or the `--threshold` F1 value.
