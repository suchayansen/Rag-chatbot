from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from rag.gdoc_loader import load_google_doc
from rag.chunker import chunk_text
from rag.vector_store import VectorStore
from rag.llm import generate_answer

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

store = VectorStore()
messages = []
last_topic = None
MAX_HISTORY = 5


def wants_concise_answer(query: str) -> bool:
    phrases = ["explain less", "briefly", "short answer", "in short", "summarize"]
    return any(p in query.lower() for p in phrases)


def is_ambiguous(query: str) -> bool:
    q = query.lower()
    if wants_concise_answer(q):
        return False
    vague = ["this", "that", "it", "explain", "more", "what about"]
    return len(q.split()) < 3 or any(v in q for v in vague)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "messages": messages}
    )


@app.post("/ingest", response_class=HTMLResponse)
def ingest(request: Request, doc_url: str = Form(...)):
    try:
        text = load_google_doc(doc_url)
        chunks = chunk_text(text)
        store.build(chunks)
        messages.clear()
        global last_topic
        last_topic = None

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "messages": messages,
                "ingest_status": "Document ingested successfully.",
                "ingest_type": "success"
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "messages": messages,
                "ingest_status": str(e),
                "ingest_type": "error"
            }
        )


@app.post("/chat", response_class=HTMLResponse)
def chat(request: Request, query: str = Form(...)):
    global last_topic

    if store.index is None:
        messages.append({"role": "bot", "content": "Please ingest a document first."})
        return templates.TemplateResponse(
            "index.html", {"request": request, "messages": messages}
        )

    messages.append({"role": "user", "content": query})

    concise = wants_concise_answer(query)

    if is_ambiguous(query) and not last_topic:
        messages.append({
            "role": "bot",
            "content": "Could you please clarify your question?"
        })
        return templates.TemplateResponse(
            "index.html", {"request": request, "messages": messages}
        )

    follow_up_words = ["explain more", "what about", "tell me more", "continue"]
    if any(w in query.lower() for w in follow_up_words) and last_topic:
        standalone_query = f"{last_topic} - {query}"
    else:
        standalone_query = query

    retrieved = store.search(standalone_query)

    if not retrieved:
        answer = "This info isn't in the document."
    else:
        context = "\n\n".join(retrieved)
        answer = generate_answer(
            standalone_query,
            context,
            concise=concise
        )
        last_topic = standalone_query

    messages.append({"role": "bot", "content": answer})

    if len(messages) > MAX_HISTORY * 2:
        messages[:] = messages[-MAX_HISTORY * 2:]

    return templates.TemplateResponse(
        "index.html", {"request": request, "messages": messages}
    )
