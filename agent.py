import ollama
import json
import os
import re
import requests
import uuid
import logging
import base64
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
import secrets

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
save_executor = ThreadPoolExecutor(max_workers=4)

ACCESS_TOKEN = secrets.token_urlsafe(16)
DEFAULT_MODEL = "llama3.1"
# Especificamos el host explícitamente para evitar fallos de resolución
client = ollama.AsyncClient(host="http://127.0.0.1:11434")

class EliteAgent:
    def __init__(self, model=DEFAULT_MODEL, memory_file="memory.json", sessions_dir="sessions"):
        self.model = model
        self.memory_file = Path(memory_file)
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.memory = self.load_memory()
        self.stop_requested = False

    def load_memory(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for k in ["sessions", "essential", "categories"]:
                        if k not in data: data[k] = [] if k != "categories" else {}
                    return data
            except: pass
        return {"essential": [], "categories": {}, "sessions": []}

    def save_memory(self):
        def _save():
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=4, ensure_ascii=False)
        save_executor.submit(_save)

    def get_session_data(self, session_id):
        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: pass
        return {"history": [], "active_categories": []}

    def save_session_data(self, session_id, data):
        def _save():
            path = self.sessions_dir / f"{session_id}.json"
            # Limpiamos imágenes base64 antes de guardar para no saturar el disco
            clean_history = []
            for m in data.get("history", []):
                msg = m.copy()
                if "images" in msg: del msg["images"]
                clean_history.append(msg)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({"history": clean_history, "active_categories": data.get("active_categories", [])}, f, indent=4, ensure_ascii=False)
        save_executor.submit(_save)

    def build_prompt(self, session_data, ext=""):
        essential = "\n".join([f"- {f}" for f in self.memory["essential"]])
        cats = ", ".join(self.memory["categories"].keys())
        active = session_data.get("active_categories", [])
        active_str = "".join([f"\n[CAT {c.upper()}]: " + ", ".join(self.memory["categories"][c]) for c in active if c in self.memory["categories"]])
        
        return (
            "Eres Kym, un asistente elite de IA. Markdown SIEMPRE.\n"
            f"CORE: {essential}\nCATS: {cats}\n{active_str}\n"
            f"--- INFO EXTRA ---\n{ext}\n"
            "COMANDOS: [SEARCH: consulta], [SAVE: cat | info], [LOAD: cat]"
        )

app = FastAPI()
agent = EliteAgent()

def is_authorized(request: Request):
    if os.getenv("IS_COLAB") == "true":
        return request.query_params.get("token") == ACCESS_TOKEN
    return True

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root(request: Request):
    if not is_authorized(request):
        return {"error": f"Usa token: {ACCESS_TOKEN}"}
    return FileResponse('static/index.html')

@app.get("/favicon.ico")
async def favicon():
    return FileResponse('static/favicon.svg', media_type='image/svg+xml')

@app.get("/models")
async def get_models(request: Request):
    if not is_authorized(request): return []
    try:
        m = await client.list()
        return [x['name'] for x in m['models']]
    except: return [DEFAULT_MODEL]

@app.get("/sessions")
async def list_sess(request: Request):
    if not is_authorized(request): return []
    return agent.memory["sessions"]

@app.post("/sessions/new")
async def new_sess(id: str = Form(...), title: str = Form(...), is_ephemeral: bool = Form(False)):
    if not is_ephemeral:
        agent.memory["sessions"].insert(0, {"id": id, "title": title})
        agent.save_memory()
    return {"ok": True}

@app.post("/sessions/update")
async def upd_sess(id: str = Form(...), title: str = Form(...)):
    for s in agent.memory["sessions"]:
        if s["id"] == id:
            s["title"] = title
            agent.save_memory()
            return {"ok": True}
    return {"err": "404"}

@app.post("/sessions/load")
async def load_sess(id: str = Form(...), offset: int = Form(0)):
    data = agent.get_session_data(id)
    h = data.get("history", [])
    start = max(0, len(h) - offset - 10)
    return {"history": h[start:(len(h)-offset)], "has_more": start > 0}

@app.post("/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    model: str = Form(DEFAULT_MODEL),
    file: UploadFile = File(None)
):
    if not is_authorized(request): raise HTTPException(401)
    
    agent.stop_requested = False
    session_data = agent.get_session_data(session_id)
    file_ctx = ""
    imgs = []

    if file:
        raw = await file.read()
        if file.content_type.startswith("image/"):
            imgs.append(base64.b64encode(raw).decode('utf-8'))
            file_ctx = f"[Imagen: {file.filename}]"
        elif file.filename.endswith(".pdf"):
            from io import BytesIO
            pdf = PdfReader(BytesIO(raw))
            file_ctx = f"--- PDF {file.filename} ---\n" + "\n".join([p.extract_text() for p in pdf.pages])[:4000]
        else:
            file_ctx = f"--- ARCHIVO {file.filename} ---\n" + raw.decode('utf-8', errors='ignore')[:4000]

    # Añadir mensaje de usuario con imágenes si las hay
    user_msg = {"role": "user", "content": message}
    if imgs:
        user_msg["images"] = imgs
    session_data["history"].append(user_msg)

    async def generate():
        ctx = file_ctx
        try:
            print(f"--- Chat con {model} ---")
            yield "status:thinking:Kym está analizando..."
            
            # Razonamiento (Incluimos las imágenes en el historial que enviamos)
            full_messages = [{"role": "system", "content": agent.build_prompt(session_data, ctx)}] + session_data["history"]
            
            res = await client.chat(model=model, messages=full_messages)
            thought = res['message']['content']
            search = re.search(r"\[SEARCH:\s*(.*?)\]", thought)
            if search:
                query = search.group(1)
                yield f"status:network:Buscando en internet: {query}..."
                print(f"--- Buscando: {query} ---")
                # En las versiones nuevas de ddg, el parámetro es 'keywords'
                s_res = await asyncio.to_thread(DDGS().text, keywords=query, max_results=3)
                ctx += f"\n[Resultados Web]: {s_res}"
                # Actualizar el system prompt con los nuevos datos
                full_messages[0]["content"] = agent.build_prompt(session_data, ctx)


            # Respuesta Final Streaming
            yield "status:writing:Redactando respuesta..."
            full_ans = ""
            async for chunk in await client.chat(model=model, messages=full_messages, stream=True):
                if agent.stop_requested: break
                text = chunk['message']['content']
                full_ans += text
                yield re.sub(r"\[.*?\]", "", text)

            # Guardar en memoria y persistir
            clean = re.sub(r"\[.*?\]", "", full_ans).strip()
            saves = re.findall(r"\[SAVE:\s*(.*?)\s*\|\s*(.*?)\]", full_ans)
            if saves:
                yield "status:saving:Memorizando..."
                for c, f in saves:
                    c, f = c.strip().lower(), f.strip()
                    if c == "essential": agent.memory["essential"].append(f)
                    else:
                        if c not in agent.memory["categories"]: agent.memory["categories"][c] = []
                        agent.memory["categories"][c].append(f)
                agent.save_memory()

            session_data["history"].append({"role": "assistant", "content": clean})
            agent.save_session_data(session_id, session_data)
            print("--- Listo ---")

        except Exception as e:
            logger.error(f"Error: {e}")
            yield f"\n⚠️ Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/stop")
async def stop(): agent.stop_requested = True; return {"ok": True}

import socket
import subprocess

def ensure_ollama_running():
    """Comprueba si Ollama está activo, si no, intenta iniciarlo."""
    port = 11434
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', port)) != 0:
            print("⚠️ Ollama no detectado. Intentando iniciar servicio automáticamente...")
            try:
                # Iniciar ollama serve en segundo plano
                subprocess.Popen(["ollama", "serve"], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                
                # Esperar hasta 10 segundos a que el puerto se abra
                for _ in range(10):
                    time.sleep(1)
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s2:
                        if s2.connect_ex(('127.0.0.1', port)) == 0:
                            print("✅ Servicio Ollama iniciado con éxito.")
                            return True
                print("❌ No se pudo iniciar Ollama automáticamente. Por favor, ábrelo manualmente.")
            except Exception as e:
                print(f"❌ Error al intentar ejecutar Ollama: {e}")
        else:
            print("🚀 Ollama ya está en ejecución.")
    return True

if __name__ == "__main__":
    import uvicorn
    import time
    
    # Asegurar que el cerebro esté encendido
    ensure_ollama_running()
    
    if os.getenv("IS_COLAB") == "true":
        print(f"\n🔑 TOKEN: {ACCESS_TOKEN}\n")
    else:
        from threading import Timer
        import webbrowser
        Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
