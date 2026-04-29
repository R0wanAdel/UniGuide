
import json
import re
import os

def clean_arabic_text(text):
    """
    دالة لتنظيف وتوحيد النصوص العربية (Normalization)
    دا مه جداً في مشاريع الـ NLP عشان النموذج يفهم الكلام صح.
    """
    # 1. إزالة علامة التطويل (Tatweel) "ـ"
    text = re.sub(r'\u0640', '', text)
    
    # 2. توحيد حرف الألف (أ، إ، آ -> ا)
    text = re.sub(r'[أإآ]', 'ا', text)
    
    # 3. توحيد حرف الياء (ى -> ي)
    text = re.sub(r'ى', 'ي', text)
    
    # --- (جديد) إزالة الرموز المتكررة ---
    
    # 4. إزالة النقاط المتكررة (.. أو ...) واستبدالها بمسافة
    text = re.sub(r'\.{2,}', ' ', text)
    
    # 5. إزالة الشرطات المتكررة (--- أو ----) واستبدالها بمسافة
    text = re.sub(r'-{2,}', ' ', text)
    
    # 6. إزالة العلامة | المتكررة (||) واستبدالها بمسافة
    text = re.sub(r'\|{2,}', ' ', text)
    
    # ------------------------------------
    
    # 7. تنظيف المسافات الزائدة والأسطر الجديدة المتكررة
    # هنا بنضيف التعديلات اللي فاتت ونعمل مسافة واحدة بس
    text = re.sub(r'\s+', ' ', text)
    
    # إزالة المسافات من البداية والنهاية
    text = text.strip()
    
    return text

def preprocess_data():
    # أسماء الملفات
    input_file = "extracted_bylaws.txt"
    output_file = "preprocessed_data.json"
    
    # التأكد من وجود ملف الاستخراج الأول
    if not os.path.exists(input_file):
        print(f"❌ خطأ: ملف {input_file} مش موجود! اتأكد إنك شغلت كود app.py الأول.")
        return

    print("🚀 جاري قراءة الملف المستخرج...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print("✂️ جاري تقسيم النص حسب الصفحات...")
    # بنستخدم الـ Regex عشان نفصل النص بناءً على السطر اللي حاطه زميلك: --- Page X ---
    # الـ capturing group (\d+) بيخليك تأخد رقم الصفحة
    parts = re.split(r'--- Page (\d+) ---', content)

    data = []
    
    # الـ parts هيبقى ليست: ['', '1', 'نص الصفحة 1', '2', 'نص الصفحة 2', ...]
    # فبنمشي عليهم خطوتين خطوتين (رقم الصفحة وبعدين النص)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            page_num = int(parts[i])
            raw_text = parts[i+1]
            
            # تطبيق التنظيف على النص
            cleaned_text = clean_arabic_text(raw_text)
            
            # بنضيف الصفحة للداتا لو النص مش فاضي بعد التنظيف
            if len(cleaned_text) > 5: 
                data.append({
                    "page_number": page_num,
                    "content": cleaned_text
                })

    print(f"✅ تمت معالجة {len(data)} صفحة بنجاح.")
    
    # الحفظ في ملف JSON
    print("💾 جاري الحفظ في ملف JSON...")
    with open(output_file, 'w', encoding='utf-8') as f:
        # ensure_ascii=False مهم جداً عشان يظهر العربي في الملف بشكل صحيح مش رموز
        # indent=4 عشان التنسيق يبقى حلو وقابل للقراءة
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"🎉 تم الانتهاء! تم حفظ الملف باسم: {output_file}")

if __name__ == "__main__":
    preprocess_data()