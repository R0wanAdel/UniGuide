import re
import json
import pyarabic.araby as araby


def clean_arabic_text(text):

    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)

    # 2. توحيد الحروف
    text = re.sub(r"[أإآ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ة", "ه", text)

    text = re.sub(r"[^a-zA-Z0-9 \n\u0600-\u06FF]", " ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()

    return text


def structural_mapping(raw_text):

    start_idx = raw_text.find("--- Page 5 ---")
    if start_idx != -1:
        raw_text = raw_text[start_idx:]

    cleaned_text = clean_arabic_text(raw_text)

    split_patterns = r"\n\s*(رؤيه الكليه|رساله الكليه|منهجيه اعداد اللائحه|ماده \d+.*)"
    parts = re.split(split_patterns, cleaned_text)

    chunks = []
    chunk_id = 1

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip() if (i + 1) < len(parts) else ""

        content = re.sub(r"\n+", " ", content)

        if content or title:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "title": title[:50],
                    "content": f"{title}: {content}".strip(),
                    "metadata": {"source": "كلية-الحاسبات-والمعلومات.pdf"},
                }
            )
            chunk_id += 1

    return chunks


def process_pipeline(input_txt_path, output_json_path):
    print("1. جاري قراءة الملف الخام...")
    with open(input_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("2. جاري التنظيف والتقسيم الهيكلي الذكي...")
    json_data = structural_mapping(raw_text)

    print("3. جاري حفظ البيانات المعالجة في ملف JSON...")
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

    print(
        f"\nتمت العملية بنجاح! تم تجهيز {len(json_data)} مقطع حقيقي ونظيف في: {output_json_path}"
    )


input_file = "extracted_bylaws.txt"
output_file = "preprocessed_data.json"
process_pipeline(input_file, output_file)
