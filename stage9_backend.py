from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uvicorn
import requests
import stage7_semantic_search as engine

OPENROUTER_API_KEY = (
    "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"
)


app = FastAPI(title="UniGuide Smart API")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
async def read_index():
    # ده بيرجع ملف الـ HTML بتاعك لليوزر
    return FileResponse('static/index.html')


@app.on_event("startup")
async def load_engine():
    global DATA, TEXTS, MODEL, EMBEDDINGS
    print("🔄 Loading Semantic Search Engine...")
    DATA, TEXTS = engine.load_chunks("preprocessed_data.json")
    MODEL = engine.SentenceTransformer(engine.MODEL_NAME)
    EMBEDDINGS = engine.load_or_build_embeddings(MODEL, TEXTS, "semantic_index.npz")
    print("✅ System Ready!")


def call_openrouter_llm(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {
                "role": "system",
                "content": "أنت مساعد ذكي لطلاب كلية الحاسبات والمعلومات. أجب عن سؤال الطالب بناءً على نصوص اللائحة المقدمة لك فقط. إذا لم تكن الإجابة في النصوص، قل 'لا أعرف'.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        data = response.json()
        if response.status_code == 200:
            return data["choices"][0]["message"]["content"]
        else:
            return f"OpenRouter Error: {data}"
    except Exception as e:
        return f"Connection Error: {str(e)}"


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="السؤال فارغ")

    # 1. البحث عن النصوص
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

    prompt = f"""النصوص المستخرجة:
{context_text}

سؤال الطالب: {request.question}
الإجابة:"""

    final_answer = call_openrouter_llm(prompt)

    return QueryResponse(
        question=request.question,
        final_answer=final_answer.strip(),
        chunks_used=formatted_results,
    )


if __name__ == "__main__":
    uvicorn.run("stage9_backend:app", host="127.0.0.1", port=8080, reload=True)
