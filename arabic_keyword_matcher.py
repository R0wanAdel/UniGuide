import argparse
import json
import re
import sys
from collections import Counter
from typing import Dict, Iterable, List, Sequence

try:
    import pyarabic.araby as araby
except ModuleNotFoundError:
    araby = None


ARABIC_LETTERS = re.compile(r"[\u0600-\u06FF]+")
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

STOPWORDS = {
    "في",
    "من",
    "عن",
    "على",
    "علي",
    "الى",
    "الي",
    "او",
    "و",
    "ما",
    "هو",
    "هي",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "كل",
    "يتم",
    "تم",
}

PREFIXES = (
    "والل",
    "فالل",
    "بالل",
    "كالل",
    "ولل",
    "فلل",
    "بلل",
    "كلل",
    "لل",
    "وال",
    "فال",
    "بال",
    "كال",
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
)

SUFFIXES = (
    "كما",
    "هما",
    "كم",
    "كن",
    "نا",
    "ها",
    "هم",
    "هن",
    "ات",
    "ون",
    "ين",
    "ان",
    "ية",
    "يه",
    "ه",
    "ة",
    "ي",
    "ك",
    "ا",
)

CANONICAL_TERMS = {
    "نظا": "نظام",
    "طلاب": "طالب",
    "طلبه": "طالب",
    "طلبة": "طالب",
    "طالبه": "طالب",
    "طالبات": "طالب",
}

OCR_CORRECTIONS = {
    "نظا": "نظام",
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text before keyword matching."""
    if araby:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    else:
        text = DIACRITICS.sub("", text)
        text = text.replace("ـ", "")
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"ة", "ه", text)
    return text


def fix_common_ocr_errors(text: str) -> str:
    """Fix very common OCR misses for display and matching."""
    for wrong, correct in OCR_CORRECTIONS.items():
        text = re.sub(rf"\b{wrong}\b", correct, text)
    return text


def tokenize(text: str) -> List[str]:
    normalized = fix_common_ocr_errors(normalize_arabic(text))
    return ARABIC_LETTERS.findall(normalized)


def strip_prefix(word: str) -> str:
    for prefix in PREFIXES:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            return word[len(prefix) :]
    return word


def strip_suffix(word: str) -> str:
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def light_stem(word: str) -> str:
    """Apply conservative Arabic light stemming for baseline retrieval."""
    word = normalize_arabic(word)
    word = CANONICAL_TERMS.get(word, word)

    stem = strip_prefix(word)
    stem = strip_suffix(stem)
    stem = CANONICAL_TERMS.get(stem, stem)

    return stem


def stem_tokens(text: str) -> List[str]:
    return [
        stem
        for stem in (light_stem(token) for token in tokenize(text))
        if stem and stem not in STOPWORDS and len(stem) > 1
    ]


def build_document_vector(text: str) -> Counter:
    return Counter(stem_tokens(text))


class ArabicKeywordMatcher:
    """Simple baseline search over bylaw chunks using Arabic keyword overlap."""

    def __init__(self, documents: Sequence[Dict]):
        self.documents = list(documents)
        self.index = [
            build_document_vector(f"{doc.get('title', '')} {doc.get('content', '')}")
            for doc in self.documents
        ]

    @classmethod
    def from_json(cls, json_path: str) -> "ArabicKeywordMatcher":
        with open(json_path, "r", encoding="utf-8") as file:
            documents = json.load(file)
        return cls(documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_terms = build_document_vector(query)
        if not query_terms:
            return []

        results = []
        for document, document_terms in zip(self.documents, self.index):
            matched_terms = sorted(set(query_terms) & set(document_terms))
            if not matched_terms:
                continue

            score = sum(query_terms[term] * document_terms[term] for term in matched_terms)
            results.append(
                {
                    "score": score,
                    "matched_terms": matched_terms,
                    "chunk": document,
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def format_result(result: Dict) -> str:
    chunk = result["chunk"]
    title = fix_common_ocr_errors(chunk.get("title", ""))
    content = fix_common_ocr_errors(chunk.get("content", ""))
    preview = content[:260] + ("..." if len(content) > 260 else "")
    return (
        f"Score: {result['score']}\n"
        f"Chunk ID: {chunk.get('chunk_id')}\n"
        f"Title: {title}\n"
        f"Matched terms: {', '.join(result['matched_terms'])}\n"
        f"Preview: {preview}"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Baseline Arabic keyword matcher with light stemming."
    )
    parser.add_argument("query", help="Arabic question or keywords to search for.")
    parser.add_argument(
        "--data",
        default="preprocessed_data.json",
        help="Path to the preprocessed chunks JSON file.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to show.")
    args = parser.parse_args()

    matcher = ArabicKeywordMatcher.from_json(args.data)
    results = matcher.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching chunks found.")
        return

    for index, result in enumerate(results, start=1):
        print(f"\nResult #{index}")
        print(format_result(result))


if __name__ == "__main__":
    main()
