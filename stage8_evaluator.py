"""
Stage 8 - Arabic Evaluator

Tracks Accuracy, F1 Score, and Exact Match for generated answers while
accounting for common Arabic spelling variants and synonyms.
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import pyarabic.araby as araby
except ModuleNotFoundError:
    araby = None


DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN_RE = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]+")

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

STOPWORDS = {
    "في",
    "من",
    "عن",
    "على",
    "علي",
    "الى",
    "الي",
    "او",
    "أو",
    "و",
    "ف",
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
    "ان",
    "أن",
    "إن",
}

PREFIXES = (
    "وال",
    "فال",
    "بال",
    "كال",
    "لل",
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
)

SUFFIXES = (
    "هما",
    "كما",
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

SYNONYM_GROUPS = (
    ("طالب", "طلاب", "طلبه", "طلبة", "دارس", "متعلم"),
    ("كليه", "كلية"),
    ("جامعه", "جامعة"),
    ("لائحه", "لائحة", "نظام", "قانون", "لوائح"),
    ("ساعه", "ساعة", "ساعات"),
    ("معتمد", "معتمده", "معتمدة", "اعتماد"),
    ("فصل", "فصول", "ترم", "ترمات"),
    ("دراسي", "دراسيه", "دراسية"),
    ("مقرر", "مقررات", "ماده", "مادة", "مواد", "كورسات"),
    ("تسجيل", "يسجل", "سجل", "قيد", "يقيد", "التحاق"),
    ("قبول", "يقبل", "التحاق"),
    ("تخرج", "التخرج", "يتخرج"),
    ("درجه", "درجة", "درجات", "بكالوريوس"),
    ("قسم", "اقسام", "أقسام", "تخصص", "تخصصات", "برنامج", "برامج"),
    ("حاسب", "حاسبات", "كمبيوتر", "الحاسوب"),
    ("ذكاء", "الذكاء"),
    ("اصطناعي", "اصطناعى", "الصناعي"),
    ("ادني", "ادنى", "أدنى", "اقل", "أقل", "الحد الادني", "الحد الادنى"),
    ("اقصي", "اقصى", "أقصى", "اكثر", "أكثر", "الحد الاقصي", "الحد الاقصى"),
    ("نجاح", "ناجح", "يجتاز", "اجتياز"),
    ("رسوب", "راسب", "يفشل"),
)

CANONICAL_TERMS: Dict[str, str] = {}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fair matching."""
    text = str(text or "").translate(ARABIC_DIGITS)

    if araby:
        text = araby.strip_tashkeel(text)
        text = araby.strip_tatweel(text)
    else:
        text = DIACRITICS.sub("", text)
        text = text.replace("ـ", "")

    text = re.sub(r"[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


for group in SYNONYM_GROUPS:
    canonical = normalize_arabic(group[0])
    for term in group:
        CANONICAL_TERMS[normalize_arabic(term)] = canonical


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


def canonicalize_token(token: str) -> str:
    token = normalize_arabic(token)
    token = CANONICAL_TERMS.get(token, token)
    token = strip_suffix(strip_prefix(token))
    return CANONICAL_TERMS.get(token, token)


def tokenize(text: str) -> List[str]:
    normalized = normalize_arabic(text)
    tokens = []
    for token in TOKEN_RE.findall(normalized):
        canonical = canonicalize_token(token)
        if canonical and canonical not in STOPWORDS and len(canonical) > 1:
            tokens.append(canonical)
    return tokens


def canonical_text(text: str) -> str:
    return " ".join(tokenize(text))


def exact_match(prediction: str, reference: str) -> float:
    return float(canonical_text(prediction) == canonical_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)

    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def as_references(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def get_first(record: Dict, keys: Sequence[str], default=None):
    for key in keys:
        if key in record:
            return record[key]
    return default


def score_record(record: Dict, threshold: float) -> Dict:
    prediction = str(
        get_first(record, ("prediction", "predicted_answer", "answer", "generated_answer"), "")
    )
    references = as_references(
        get_first(record, ("references", "reference", "gold", "ground_truth", "expected_answer"), [])
    )

    if not references:
        raise ValueError("Each record must include a reference/gold/ground_truth answer.")

    pairs: List[Tuple[float, float, str]] = [
        (exact_match(prediction, reference), f1_score(prediction, reference), reference)
        for reference in references
    ]
    best_em, best_f1, best_reference = max(pairs, key=lambda item: (item[0], item[1]))

    expected_chunk_id = get_first(record, ("expected_chunk_id", "gold_chunk_id"))
    predicted_chunk_id = get_first(record, ("predicted_chunk_id", "retrieved_chunk_id"))
    retrieval_correct = None
    if expected_chunk_id is not None and predicted_chunk_id is not None:
        retrieval_correct = str(expected_chunk_id) == str(predicted_chunk_id)

    answer_correct = bool(best_em or best_f1 >= threshold)
    accuracy = float(retrieval_correct if retrieval_correct is not None else answer_correct)

    return {
        "question": get_first(record, ("question", "query"), ""),
        "prediction": prediction,
        "best_reference": best_reference,
        "exact_match": best_em,
        "f1": best_f1,
        "accuracy": accuracy,
        "retrieval_correct": retrieval_correct,
    }


def load_records(path: str) -> List[Dict]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    if suffix == ".csv":
        with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("samples", data.get("data", []))

    if not isinstance(data, list):
        raise ValueError("Evaluation file must be a list, or a dict with samples/data.")

    return data


def evaluate(records: Iterable[Dict], threshold: float = 0.8) -> Dict:
    details = [score_record(record, threshold) for record in records]
    total = len(details)

    if total == 0:
        return {"total": 0, "accuracy": 0.0, "f1": 0.0, "exact_match": 0.0, "details": []}

    return {
        "total": total,
        "accuracy": sum(item["accuracy"] for item in details) / total,
        "f1": sum(item["f1"] for item in details) / total,
        "exact_match": sum(item["exact_match"] for item in details) / total,
        "details": details,
    }


def print_report(report: Dict, show_details: bool) -> None:
    print("Stage 8 Evaluation Report")
    print("=" * 28)
    print(f"Samples: {report['total']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"F1 Score: {report['f1']:.4f}")
    print(f"Exact Match: {report['exact_match']:.4f}")

    if show_details:
        for index, item in enumerate(report["details"], start=1):
            print(f"\nSample #{index}")
            print(f"Question: {item['question']}")
            print(f"Accuracy: {item['accuracy']:.4f}")
            print(f"F1: {item['f1']:.4f}")
            print(f"Exact Match: {item['exact_match']:.4f}")
            print(f"Prediction: {item['prediction']}")
            print(f"Best Reference: {item['best_reference']}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Stage 8 Arabic answer evaluator.")
    parser.add_argument("eval_file", help="JSON, JSONL, or CSV file containing predictions and references.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Minimum synonym-aware F1 needed to count an answer as accurate.",
    )
    parser.add_argument("--details", action="store_true", help="Print per-sample scores.")
    parser.add_argument("--output-json", help="Optional path to save the full report as JSON.")
    args = parser.parse_args()

    records = load_records(args.eval_file)
    report = evaluate(records, threshold=args.threshold)
    print_report(report, show_details=args.details)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
