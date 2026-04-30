import json
from sentence_transformers import SentenceTransformer, util

class TransformerEngine:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        # تحميل نموذج يدعم العربية بكفاءة
        self.model = SentenceTransformer(model_name)
        self.data = []
        self.embeddings = None

    def load_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f) # يفترض أن الـ json يحتوي على نصوص اللوائح
        
        # تحويل النصوص إلى Embeddings
        texts = [item['text'] for item in self.data]
        self.embeddings = self.model.encode(texts, convert_to_tensor=True)
        print("Embeddings generated successfully.")

    def search(self, query, top_k=3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # البحث عن أكثر الفقرات تشابهاً في المعنى
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)
        
        results = []
        for hit in hits[0]:
            results.append(self.data[hit['corpus_id']])
        return results

# تجربة سريعة
if __name__ == "__main__":
    engine = TransformerEngine()
    engine.load_data('preprocessed_data.json')
    query = "ما هي عقوبة التأخير في تسليم المشروع؟"
    results = engine.search(query)
    print(results)