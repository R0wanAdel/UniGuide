from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import google.generativeai as genai
import stage7_semantic_search as engine

GEMINI_API_KEY = "AIzaSyAU3bJ4VdjmkssiGz_ZJ5w2C8P_ooFs4KU"
genai.configure(api_key=GEMINI_API_KEY)


best_model_name = ""
try:
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:

            if "gemini" in m.name:
                best_model_name = m.name.replace("models/", "")
                break
except Exception as e:
    best_model_name = "gemini-1.5-flash"

print(f" AI Model Successfully Selected: {best_model_name}")
llm_model = genai.GenerativeModel(best_model_name)


app = FastAPI(title="UniGuide Smart API")

DATA = None
TEXTS = None
MODEL = None
EMBEDDINGS = None


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    question: str
    final_answer: str
    chunks_used: list


@app.on_event("startup")
async def load_engine():
    global DATA, TEXTS, MODEL, EMBEDDINGS
    print("🔄 Loading Semantic Search Engine...")
    DATA, TEXTS = engine.load_chunks("preprocessed_data.json")
    MODEL = engine.SentenceTransformer(engine.MODEL_NAME)
    EMBEDDINGS = engine.load_or_build_embeddings(MODEL, TEXTS, "semantic_index.npz")
    print("✅ System Ready!")


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="السؤال فارغ")

    search_results = engine.search(
        query=request.question,
        data=DATA,
        texts=TEXTS,
        embeddings=EMBEDDINGS,
        model=MODEL,
        top_k=request.top_k,
    )

    formatted_results = []
    context_text = ""
    for res in search_results:
        chunk = DATA[res["index"]]
        content = chunk.get("content", "")
        score = res.get("score", res.get("similarity", 0.0))

        formatted_results.append(
            {
                "title": chunk.get("title", "بدون عنوان"),
                "content": content,
                "score": round(float(score), 4),
            }
        )
        context_text += content + "\n\n"

    prompt = f"""
    أنت مساعد ذكي لطلاب كلية الحاسبات والمعلومات.
    أجب عن سؤال الطالب بناءً على نصوص اللائحة التالية فقط.
    إذا لم تكن الإجابة موجودة في النصوص، قل "عذراً، لا أملك معلومات حول هذا السؤال من اللائحة".
    الرجاء صياغة الإجابة بلغة عربية سليمة وواضحة، وتصحيح أي أخطاء إملائية.

    النصوص المستخرجة:
    {context_text}

    سؤال الطالب: {request.question}
    الإجابة:
    """

    try:
        response = llm_model.generate_content(prompt)
        final_answer = response.text
    except Exception as e:
        final_answer = f"حدث خطأ أثناء توليد الإجابة: {str(e)}"

    return QueryResponse(
        question=request.question,
        final_answer=final_answer.strip(),
        chunks_used=formatted_results,
    )


if __name__ == "__main__":
    uvicorn.run("stage9_backend:app", host="127.0.0.1", port=8000, reload=True)
