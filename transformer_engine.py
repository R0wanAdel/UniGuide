import json
import torch
import os
from sentence_transformers import SentenceTransformer, util


class TransformerEngine:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.data = []
        self.embeddings = None
        self.texts = []
        self.embeddings_path = "embeddings.pt"

    def load_data(self, json_path):

        if not os.path.exists(json_path):
            print(f"❌ Error: {json_path} not found!")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.texts = [f"{item['title']}: {item['content']}" for item in self.data]

        if os.path.exists(self.embeddings_path):
            print(" Loading saved embeddings...")
            self.embeddings = torch.load(self.embeddings_path, map_location=self.device)
        else:
            print(" Generating new embeddings... This might take a minute.")
            self.embeddings = self.model.encode(self.texts, convert_to_tensor=True)
            torch.save(self.embeddings, self.embeddings_path)
            print(" Embeddings generated and saved.")

    def search(self, query, top_k=3, threshold=0.35):
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)

        print(f"\n🔍 نتائج البحث عن: '{query}'")
        print("=" * 50)

        results = []
        for hit in hits[0]:

            if hit["score"] < threshold:
                continue

            corpus_id = hit["corpus_id"]
            score = hit["score"]
            item = self.data[corpus_id]

            print(f" العنوان: {item['title']}")
            print(f" درجة التشابه: {score:.4f}")
            print(f" النص: {item['content'][:200]}...")
            print("-" * 30)

            results.append(item)

        if not results:
            print(" لم يتم العثور على نتائج مطابقة بدقة كافية.")

        return results


if __name__ == "__main__":
    import sys

    engine = TransformerEngine()

    engine.load_data("preprocessed_data.json")

    user_query = sys.argv[1] if len(sys.argv) > 1 else "ما هي شروط القبول بالكلية؟"

    engine.search(user_query)
