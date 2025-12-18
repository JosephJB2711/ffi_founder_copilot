from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from uuid import uuid4
import requests
import chromadb
import os

from session_store import (
    init_db,
    touch_session,
    get_summary,
    set_summary,
    append_message,
    get_last_messages,
    count_messages,
    delete_oldest_messages,
)

# ─────────────────────────────
# PATHS (absolut, konsistent)
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Adresse deines lokal laufenden Ollama-Servers
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
MODEL_NAME = "llama3.2:latest"

# Strenger Prompt (wenn RAG vorhanden)
SYSTEM_PROMPT_STRICT = """
Du bist der FFI Founder Copilot – der offizielle, kritische Sparringspartner und Umsetzungsassistent der Future Founders Initiative e.V. (FFI).

Deine Hauptaufgabe:
Du unterstützt Nutzer:innen bei der Planung, Strukturierung und Umsetzung von FFI-Projekten, Events, Sponsoring-Aktivitäten, Orga-Themen, Community-Building und Founder-Ideen. Du bestätigst keine Aussagen blind, sondern prüfst sie kritisch, hinterfragst Annahmen und machst Vorschläge, wie etwas besser, klarer und wirksamer umgesetzt werden kann.

Grundprinzipien deines Verhaltens:
1. Du bist analytisch, ehrlich und lösungsorientiert.
2. Du priorisierst Logik, Umsetzbarkeit und Klarheit über Zustimmung oder Harmonie.
3. Du hilfst, aus vagen oder chaotischen Ideen strukturierte, realistische Pläne zu machen.
4. Du arbeitest immer im Interesse der FFI-Mission: junge Menschen befähigen, unternehmerisch Verantwortung zu übernehmen.

--------------------
1. Rolle und Scope
--------------------
Du agierst als interner FFI-Copilot, nicht als externer Unternehmensberater.

Du unterstützt insbesondere in diesen Bereichen:
- Eventplanung (Formate, Abläufe, Ziele, Teilnehmererlebnis)
- Orga & Prozesse (Rollen, Verantwortlichkeiten, Kommunikation)
- Sponsoring & Partner Outreach (Wertversprechen, Mails, Follow-Ups)
- Founder-Ideenentwicklung (Strukturierung, Schärfung, Roadmaps)
- Community Building (Formate, Engagement, Bindung)
- interne Kommunikation (Mails, Texte, Beschreibungen, Pitch-Material)
- Nutzung und Umsetzung interner FFI-Playbooks, Guidelines und Dokumente
- Legal-Themen nur insoweit, wie sie sich aus FFI-internen Materialien (z. B. Legal Event Guide, Datenschutz, Event Terms) ergeben – keine eigenständige Rechtsberatung außerhalb dieser Basis.

Du bist kein: 
- Ersatz für einen Rechtsanwalt außerhalb der FFI-Dokumente,
- generischer Motivationscoach,
- beliebiger Marketing-Bot.

--------------------
2. Umgang mit Wissensbasis (RAG)
--------------------
Wenn eine Wissensbasis / Dokumente (z. B. Event Terms, Legal Event Guide, Datenschutz-Richtlinien, Sponsoring-Template, Orga-Notizen, vergangene Event-Auswertungen) verfügbar sind, gehst du wie folgt vor:

1. Du versuchst immer zuerst, die Antwort aus diesen Dokumenten abzuleiten.
2. Du verweist inhaltlich auf relevante Teile („In den Event Terms wird geregelt, dass…“, „Im Legal Event Guide steht, dass…“).
3. Wenn die Wissensbasis keine klare Antwort liefert:
   - Du spekulierst nicht und erfindest keine Regeln.
   - Du machst transparent, dass die Grundlage fehlt.
   - Du schlägst vor, welche Infos oder Dokumente noch gebraucht werden.
4. Du machst klar, wenn etwas eine Empfehlung, Einschätzung oder Hypothese ist und nicht ausdrücklich in den FFI-Dokumenten steht.

--------------------
3. Kommunikationsstil und Output
--------------------
Dein Stil ist:
- klar, direkt, strukturiert
- kritisch, aber konstruktiv
- fokussiert auf Umsetzung und Qualität
- frei von unnötigen Floskeln und Übertreibungen

--------------------
6. Umgang mit Unsicherheit und Grenzen
--------------------
- Du darfst niemals Informationen erfinden.
- Wenn die Wissensbasis keine Grundlage bietet, sag: 'Dazu liegen mir keine verlässlichen Informationen vor.'
- Spekulationen sind verboten.

WICHTIG:
- Alle Antworten müssen direkt und ausschließlich aus der Wissensbasis stammen.
- Du darfst NICHT raten oder improvisieren.
- Wenn keine Grundlage existiert, sag: 'Dazu liegen mir keine verlässlichen Informationen vor.'
- Spekulationen sind verboten.
- Du antwortest nur auf Basis der folgenden Dokumentpassagen:

WISSENSBASIS:
{retrieved_chunks}

NUTZERFRAGE:
{user_question}

AUFGABE:
Beantworte die Frage ausschließlich mit diesen Dokumenten.  
Wenn du bestimmte Details nicht sicher weißt, erwähne das explizit.  
Wenn die Dokumente keine klare Grundlage bieten, sag das.  
Keine Halluzinationen. Keine Erfindungen. Keine Vermutungen.
"""

# Fallback Prompt (wenn RAG leer ist): Session Memory soll helfen, aber ohne FFI-Regeln zu erfinden
SYSTEM_PROMPT_FALLBACK = """
Du bist der FFI Founder Copilot.

Es wurde KEIN passender Dokumentkontext aus der FFI-Wissensbasis gefunden.
Du sollst trotzdem helfen – aber mit klaren Regeln:

1) Nutze SESSION_MEMORY und den Chat-Verlauf, um die Situation zu verstehen.
2) Gib allgemeine Best Practices und konkrete nächste Schritte als Empfehlung.
3) Erfinde KEINE FFI-spezifischen Regeln, Policies oder rechtlichen Vorgaben.
4) Wenn FFI-Regeln relevant wären: formuliere sie als offene Punkte und frage nach dem passenden Dokument (Event Terms, Legal Event Guide, Datenschutz, Sponsoring-Template etc.).
5) Sei strukturiert: Analyse → Empfehlungen → Nächste Schritte → Offene Punkte/Dokumente.

NUTZERFRAGE:
{user_question}
"""

app = FastAPI()

# DB für Session Memory initialisieren (beim App-Start)
init_db()

# ───────────────── CORS ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # später einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────── Static Files ────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ──────────────── Vektordatenbank initialisieren ───────────────────────────
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection("ffi_founder_docs")

print("CHROMA COUNT:", collection.count())


def get_embedding(text: str) -> List[float]:
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def _has_satzung_chunks() -> bool:
    """
    Schutz gegen "leeren Filter": Wenn keine Satzungs-Chunks existieren,
    würde where={"doc_type":"satzung"} immer leer zurückgeben und Fallback triggern.
    """
    try:
        # Sehr kleine Probeabfrage, um zu prüfen, ob überhaupt etwas mit doc_type=satzung existiert.
        # Chroma akzeptiert where + n_results.
        probe = collection.query(
            query_embeddings=[[0.0]],  # Dummy; Chroma wird das u.U. ablehnen -> dann fallback unten
            n_results=1,
            where={"doc_type": "satzung"}
        )
        docs = probe.get("documents", [[]])[0]
        return bool(docs)
    except Exception:
        # Wenn Dummy-Embedding nicht akzeptiert wird, nutzen wir eine sichere Alternative:
        # Wir versuchen einfach normal zu queryen ohne where und checken die Metadaten.
        try:
            qemb = get_embedding("FFI Satzung")
            res = collection.query(query_embeddings=[qemb], n_results=5)
            metas = res.get("metadatas", [[]])[0] or []
            for m in metas:
                if (m or {}).get("doc_type") == "satzung":
                    return True
            return False
        except Exception:
            return False


_SATZUNG_PRESENT = None


def retrieve_context(user_text: str, k: int = 8) -> str:
    """
    Holt die k ähnlichsten Textstücke aus deiner FFI-Wissensbasis.
    """
    global _SATZUNG_PRESENT

    try:
        query_emb = get_embedding("FFI " + user_text)
    except Exception:
        return ""

    q = user_text.lower()

    # Heuristik: wenn User explizit Satzung sagt → filter, aber nur wenn Satzung auch wirklich vorhanden
    where = None
    if "satzung" in q:
        if _SATZUNG_PRESENT is None:
            _SATZUNG_PRESENT = _has_satzung_chunks()
        if _SATZUNG_PRESENT:
            where = {"doc_type": "satzung"}

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
        where=where
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return ""

    snippets = []
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        source = meta.get("source", "unbekannt")
        doc_type = meta.get("doc_type", "unknown")
        page = meta.get("page")
        chunk = meta.get("chunk")

        loc = []
        if page:
            loc.append(f"p.{page}")
        if chunk is not None:
            loc.append(f"chunk {chunk}")
        loc_str = ", ".join(loc)

        header = f"[{source} | {doc_type}"
        if loc_str:
            header += f" | {loc_str}"
        header += "]"

        snippets.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(snippets)


# ───────────────── Session Memory Settings ─────────────────────────────────
SUMMARY_TRIGGER_MESSAGES = 20
KEEP_LAST_MESSAGES = 12


def build_summary_prompt(existing_summary: str, old_messages: List[Dict[str, str]]) -> str:
    return f"""
Du aktualisierst eine kurze, faktenbasierte Session-Zusammenfassung.
Regeln:
- Nur Fakten/Entscheidungen/Definitionen/To-dos, keine Floskeln.
- Keine Vermutungen, nichts erfinden.
- Maximal 10 Bulletpoints.
- Wenn etwas unklar ist: weglassen.

Bisherige Zusammenfassung:
{existing_summary}

Neue Gesprächsteile (ältere Messages, die verdichtet werden sollen):
{old_messages}

Gib nur die aktualisierte Zusammenfassung als Bullet-Liste aus.
""".strip()


# ───────────────── Modelle ─────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    message: Optional[str] = None
    session_id: Optional[str] = None


# ───────────────── Routen ───────────────────────────────────────────────────
@app.get("/")
def read_root():
    return FileResponse("index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    # 0) User-Text robust bestimmen
    user_text = (req.message or "").strip()
    if not user_text and req.messages:
        user_text = (req.messages[-1].content or "").strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Keine Nachricht erhalten (message oder messages fehlt).")

    # 1) Session-ID
    session_id = (req.session_id or str(uuid4())).strip()
    touch_session(session_id)

    # 2) Session Memory IMMER laden (damit es wirklich wirkt)
    summary = get_summary(session_id)
    history = get_last_messages(session_id, limit=KEEP_LAST_MESSAGES)

    # 3) Rolling Summary ggf. aktualisieren (ebenfalls unabhängig von RAG)
    if count_messages(session_id) > SUMMARY_TRIGGER_MESSAGES:
        older = get_last_messages(session_id, limit=SUMMARY_TRIGGER_MESSAGES + KEEP_LAST_MESSAGES)
        old_part = older[:-KEEP_LAST_MESSAGES] if len(older) > KEEP_LAST_MESSAGES else []

        if old_part:
            summarizer_messages = [
                {"role": "system", "content": "Du bist ein präziser Protokollant. Du fasst strikt faktenbasiert zusammen."},
                {"role": "user", "content": build_summary_prompt(summary, old_part)},
            ]
            try:
                sum_resp = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "messages": summarizer_messages,
                        "stream": False,
                        "options": {"temperature": 0},
                    },
                    timeout=120,
                )
                sum_resp.raise_for_status()
                new_summary = sum_resp.json().get("message", {}).get("content", "").strip()
            except Exception:
                new_summary = ""

            if new_summary:
                set_summary(session_id, new_summary)
                delete_oldest_messages(session_id, keep_last=KEEP_LAST_MESSAGES)
                summary = new_summary
                history = get_last_messages(session_id, limit=KEEP_LAST_MESSAGES)

    # 4) RAG Kontext holen
    context = retrieve_context(user_text)

    # 5) Prompt wählen: strict wenn Kontext da, fallback wenn nicht
    if context.strip():
        system_prompt = SYSTEM_PROMPT_STRICT.format(
            retrieved_chunks=context,
            user_question=user_text,
        )
    else:
        system_prompt = SYSTEM_PROMPT_FALLBACK.format(
            user_question=user_text,
        )

    # 6) Nachrichten an Modell: system + session_memory + history + user
    messages = [{"role": "system", "content": system_prompt}]

    if summary.strip():
        messages.append({"role": "system", "content": "SESSION_MEMORY (faktenbasiert):\n%s" % summary})

    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2 if not context.strip() else 0,  # Fallback darf minimal kreativ sein, strict nicht
            "top_p": 1.0,
        },
    }

    # 7) Ollama call + persistieren
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        reply = data.get("message", {}).get("content")
        if reply is None:
            raise ValueError("Antwort von Ollama enthielt kein 'message.content'.")

        reply = reply.strip()

        append_message(session_id, "user", user_text)
        append_message(session_id, "assistant", reply)

        return {"session_id": session_id, "reply": reply}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Ollama-Fehler: %s" % e)
