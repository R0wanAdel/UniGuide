from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import stage7_semantic_search as engine

GEMINI_API_KEY = "AIzaSyBedAMC-7lfXvDGor4hzQv9zIHaWcMDeG8"


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


def get_best_model_name():
    """دالة لجلب أول موديل توليدي متاح ومسموح للمفتاح ده"""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    )
    try:
        response = requests.get(url)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for m in models:
                if "generateContent" in m.get(
                    "supportedGenerationMethods", []
                ) and "gemini" in m.get("name", ""):
                    return m["name"]
    except Exception as e:
        pass

    return "models/gemini-1.0-pro"


ACTIVE_MODEL = get_best_model_name()
print(f"🤖 Active Google Model: {ACTIVE_MODEL}")


def call_gemini_direct(prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/{ACTIVE_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        if response.status_code == 200:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Google API Error: {data.get('error', {}).get('message', 'Unknown Error')}"
    except Exception as e:
        return f"Connection Error: {str(e)}"


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

    final_answer = call_gemini_direct(prompt)

    return QueryResponse(
        question=request.question,
        final_answer=final_answer.strip(),
        chunks_used=formatted_results,
    )


if __name__ == "__main__":
    uvicorn.run("stage9_backend:app", host="127.0.0.1", port=8000, reload=True)
