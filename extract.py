import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import fitz  # PyMuPDF
import easyocr
import cv2
import numpy as np


def extract_and_save_all_pages(pdf_path, output_txt_path):

    reader = easyocr.Reader(["ar"], gpu=True)
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    print(f"بدأ استخراج النصوص من {total_pages} صفحة...")
    print(
        "الموضوع هياخد شوية وقت، سيبي الجهاز يخلص براحته (استغلي قوة كارت الـ RTX الخاص بيكِ)."
    )

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for page_num in range(total_pages):
            print(f"جاري معالجة صفحة {page_num + 1} من {total_pages}...")

            page = doc.load_page(page_num)

            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.h, pix.w, pix.n
            )

            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

            result = reader.readtext(img, detail=0, paragraph=True)
            page_text = "\n\n".join(result)

            f.write(f"\n--- Page {page_num + 1} ---\n")
            f.write(page_text)
            f.write("\n")

    print(f"\nممتاز! تم استخراج اللائحة بالكامل وحفظها في ملف: {output_txt_path}")


pdf_file_path = "كلية-الحاسبات-والمعلومات.pdf"

output_file = "extracted_bylaws.txt"

extract_and_save_all_pages(pdf_file_path, output_file)
