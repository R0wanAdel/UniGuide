"""
Stage 7 - Arabic Context & Embeddings Search - Short Version

Uses:
1. Sentence embeddings for semantic search.
2. Simple Arabic keyword boost for better ranking.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN_RE = re.compile(r"[\u0600-\u06FF0-9]+")

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
    "ف",
    "ما",
    "ماذا",
    "هو",
    "هي",
    "هذا",
    "هذه",
    "ذلك",
    "كل",
    "اي",
    "يمكن",
    "يكون",
    "تكون",
    "خلال",
    "بعد",
    "قبل",
}

CANONICAL = {
    "الساعات": "ساعه",
    "ساعات": "ساعه",
    "ساعة": "ساعه",
    "الساعه": "ساعه",
    "التسجيل": "تسجيل",
    "يسجل": "تسجيل",
    "مسجله": "تسجيل",
    "المسجله": "تسجيل",
    "الفصل": "فصل",
    "فصول": "فصل",
    "الدراسي": "دراسي",
    "الدراسيه": "دراسي",
    "دراسيه": "دراسي",
    "الحد": "حد",
    "الاقصي": "اقصي",
    "الاقصى": "اقصي",
    "اقصى": "اقصي",
    "الادني": "ادني",
    "الادنى": "ادني",
    "ادنى": "ادني",
    "المعتمده": "معتمد",
    "معتمده": "معتمد",
    "الطالب": "طالب",
    "طلاب": "طالب",
    "طلبه": "طالب",
    "طلبة": "طالب",
    "المقررات": "مقرر",
    "مقررات": "مقرر",
    "المقرر": "مقرر",
    "الغياب": "غياب",
    "انذار": "انذار",
    "حرمان": "حرمان",
    "التقدير": "تقدير",
    "المعدل": "gpa",
    "التراكمي": "gpa",
    "الاقسام": "قسم",
    "تخصصات": "قسم",
}


def normalize(text):
    text = str(text)
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


def light_stem(word):
    word = normalize(word)

    if not word:
        return ""

    if word in CANONICAL:
        return CANONICAL[word]

    prefixes = ["وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل"]
    suffixes = [
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
    ]

    for prefix in prefixes:
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix) :]
            break

    if word in CANONICAL:
        return CANONICAL[word]

    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break

    return CANONICAL.get(word, word)


def tokenize(text):
    tokens = []
    for word in TOKEN_RE.findall(normalize(text)):
        stem = light_stem(word)
        if stem and stem not in STOPWORDS and len(stem) > 1:
            tokens.append(stem)
    return tokens


def chunk_text(chunk):
    title = chunk.get("title", "")
    content = chunk.get("content", chunk.get("text", ""))
    return normalize(f"{title} {content}")


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("preprocessed_data.json must contain a list of chunks.")

    texts = [chunk_text(chunk) for chunk in data]
    return data, texts


def load_or_build_embeddings(model, texts, index_path, rebuild=False):
    index_file = Path(index_path)

    if index_file.exists() and not rebuild:
        saved = np.load(index_file)
        embeddings = saved["embeddings"].astype("float32")

        if len(embeddings) == len(texts):
            return embeddings

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    np.savez_compressed(index_file, embeddings=embeddings)
    return embeddings


def is_registration_hours_question(query):
    q_text = normalize(query)
    q_tokens = set(tokenize(query))

    has_hours = (
        "ساعه" in q_tokens
        or "معتمد" in q_tokens
        or "ساعه" in q_text
        or "ساعات" in q_text
        or "الساعات" in q_text
    )

    has_context = (
        "تسجيل" in q_tokens
        or "فصل" in q_tokens
        or "دراسي" in q_tokens
        or "حد" in q_tokens
        or "اقصي" in q_tokens
        or "ادني" in q_tokens
        or "عدد" in q_tokens
    )

    return has_hours and has_context


def keyword_score(query_tokens, document_tokens):
    if not query_tokens:
        return 0.0

    overlap = query_tokens & document_tokens
    return len(overlap) / len(query_tokens)


def phrase_boost(query, document, title):
    boost = 0.0
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(title))
    d_tokens = set(tokenize(document))

    boost += len(q_tokens & t_tokens) * 0.45

    important_terms = {
        "ساعه",
        "تسجيل",
        "فصل",
        "دراسي",
        "حد",
        "اقصي",
        "ادني",
        "معتمد",
        "غياب",
        "حرمان",
        "انذار",
        "gpa",
        "قسم",
    }
    boost += len(q_tokens & important_terms & d_tokens) * 0.15

    normalized_doc = normalize(document)
    if any(k in q_tokens for k in ["ساعه", "معتمد", "تسجيل"]):
        if "12" in normalized_doc and "18" in normalized_doc:
            boost += 0.50
        if "تسجيل" in normalize(title):
            boost += 0.40

    return boost


def search(query, data, texts, embeddings, model, top_k):
    query_text = normalize(query)

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")[0]

    semantic_scores = embeddings @ query_embedding

    query_tokens = set(tokenize(query))
    final_scores = []

    for i, text in enumerate(texts):
        document_tokens = set(tokenize(text))

        semantic = float(semantic_scores[i])
        lexical = keyword_score(query_tokens, document_tokens)
        boost = phrase_boost(query, text, data[i].get("title", ""))

        final_score = (semantic * 0.80) + (lexical * 0.20) + boost

        final_scores.append(
            {
                "index": i,
                "final_score": final_score,
                "semantic_score": semantic,
                "keyword_score": lexical,
                "phrase_boost": boost,
            }
        )

    final_scores.sort(key=lambda item: item["final_score"], reverse=True)
    return final_scores[:top_k]


def print_result(number, result, chunk):
    content = chunk.get("content", chunk.get("text", ""))
    preview = content[:650] + ("..." if len(content) > 650 else "")

    print(f"\nResult #{number}")
    print(f"Final Score: {result['final_score']:.4f}")
    print(f"Semantic Score: {result['semantic_score']:.4f}")
    print(f"Keyword Score: {result['keyword_score']:.4f}")
    print(f"Phrase Boost: {result['phrase_boost']:.4f}")
    print(f"Chunk ID: {chunk.get('chunk_id')}")
    print(f"Title: {chunk.get('title', '')}")
    print(f"Source: {chunk.get('metadata', {}).get('source', 'unknown')}")
    print(f"Preview: {preview}")


def main():
    parser = argparse.ArgumentParser(description="Stage 7 Arabic semantic search.")
    parser.add_argument("query", help="Arabic question.")
    parser.add_argument("--data", default="preprocessed_data.json")
    parser.add_argument("--index", default="semantic_index.npz")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data, texts = load_chunks(args.data)

    model = SentenceTransformer(MODEL_NAME)

    embeddings = load_or_build_embeddings(
        model=model,
        texts=texts,
        index_path=args.index,
        rebuild=args.rebuild_index,
    )

    results = search(
        query=args.query,
        data=data,
        texts=texts,
        embeddings=embeddings,
        model=model,
        top_k=args.top_k,
    )

    for number, result in enumerate(results, start=1):
        chunk = data[result["index"]]
        print_result(number, result, chunk)


if __name__ == "__main__":
    main()
