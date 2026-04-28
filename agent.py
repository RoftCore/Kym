import argparse
import logging
import os
import secrets
import json

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from file_utils import extract_file_context
from provider_registry import ProviderRegistry
from providers import (
    LocalAIProvider,
    NvidiaAPIProvider,
    extract_saved_facts,
    normalize_provider_error,
    strip_agent_commands,
)
from storage import AgentState

# Argumentos CLI
parser = argparse.ArgumentParser(description="Kym AI Assistant")
parser.add_argument(
    "--no-local",
    action="store_true",
    help="Deshabilitar el soporte de IA local (Ollama)",
)
parser.add_argument(
    "--ollama-host",
    default=os.getenv("KYM_OLLAMA_HOST", "http://127.0.0.1:11434"),
    help="URL del servidor Ollama local o remoto",
)
args, _ = parser.parse_known_args()

# Configuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCESS_TOKEN = secrets.token_urlsafe(16)
LOCAL_AI_ENABLED = not args.no_local
LOCAL_MODEL_ID = "llama3.1"
API_MODEL_ID = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_MODEL = LOCAL_MODEL_ID if LOCAL_AI_ENABLED else API_MODEL_ID
NVIDIA_API_KEY_ENV = "NVIDIA_AI_API_KEY"
NVIDIA_API_BASE_URL = "https://integrate.api.nvidia.com/v1"
OLLAMA_HOST = args.ollama_host

local_provider = LocalAIProvider(
    enabled=LOCAL_AI_ENABLED,
    host=OLLAMA_HOST,
    logger=logger,
)
api_provider = NvidiaAPIProvider(
    model_id=API_MODEL_ID,
    label="Nemotron 3 Super 120B (NVIDIA)",
    api_key_env=NVIDIA_API_KEY_ENV,
    base_url=NVIDIA_API_BASE_URL,
)
provider_registry = ProviderRegistry(
    local_provider=local_provider,
    default_api_provider=api_provider,
    logger=logger,
)


def get_provider_for_model(model_id: str):
    return provider_registry.get_provider_for_model(model_id)


class LocalProviderPayload(BaseModel):
    enabled: bool = True
    host: str = ""
    autostart: bool = True


class ApiProviderPayload(BaseModel):
    label: str
    model_id: str
    base_url: str
    api_key: str
    source_label: str = "API"


app = FastAPI()
agent = AgentState(model=DEFAULT_MODEL, logger=logger)


def is_authorized(request: Request):
    if os.getenv("IS_COLAB") == "true":
        return request.query_params.get("token") == ACCESS_TOKEN
    return True


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root(request: Request):
    if not is_authorized(request):
        return {"error": f"Usa token: {ACCESS_TOKEN}"}
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.svg", media_type="image/svg+xml")


@app.get("/models")
async def get_models(request: Request):
    if not is_authorized(request):
        return []

    provider_models = await provider_registry.list_models()
    from fastapi import Response
    
    models_data = [
        {"id": item.id, "label": item.label, "source": item.source}
        for item in provider_models
    ]
    
    return Response(
        content=json.dumps(models_data),
        media_type="application/json",
        headers={"X-Current-Model": agent.model}
    )


@app.get("/providers")
async def get_providers(request: Request):
    if not is_authorized(request):
        raise HTTPException(401)
    return provider_registry.summary()


@app.post("/providers/local")
async def configure_local_provider(request: Request, payload: LocalProviderPayload):
    if not is_authorized(request):
        raise HTTPException(401)

    data = provider_registry.configure_local(
        enabled=payload.enabled,
        host=payload.host,
        autostart=payload.autostart,
    )
    return {
        "ok": True,
        "local": data,
        "providers": provider_registry.summary(),
    }


@app.post("/providers/api")
async def add_api_provider(request: Request, payload: ApiProviderPayload):
    if not is_authorized(request):
        raise HTTPException(401)

    if not payload.label.strip():
        raise HTTPException(400, "El nombre de la IA es obligatorio.")
    if not payload.model_id.strip():
        raise HTTPException(400, "El model id es obligatorio.")
    if not payload.base_url.strip():
        raise HTTPException(400, "La base URL es obligatoria.")
    if not payload.api_key.strip():
        raise HTTPException(400, "La API key es obligatoria.")

    provider = provider_registry.add_api_provider(
        label=payload.label,
        model_id=payload.model_id,
        base_url=payload.base_url,
        api_key=payload.api_key,
        source_label=payload.source_label,
    )
    agent.model = provider["model_id"]
    return {
        "ok": True,
        "model": provider["model_id"],
        "provider": provider,
        "providers": provider_registry.summary(),
    }


@app.get("/sessions")
async def list_sessions(request: Request):
    if not is_authorized(request):
        return []
    return agent.memory["sessions"]


@app.post("/sessions/new")
async def new_session(
    id: str = Form(...),
    title: str = Form(...),
    is_ephemeral: bool = Form(False),
):
    if not is_ephemeral:
        agent.memory["sessions"].insert(0, {"id": id, "title": title})
        agent.save_memory()
    return {"ok": True}


@app.post("/sessions/update")
async def update_session(id: str = Form(...), title: str = Form(...)):
    for session in agent.memory["sessions"]:
        if session["id"] == id:
            session["title"] = title
            agent.save_memory()
            return {"ok": True}
    return {"err": "404"}


@app.post("/sessions/load")
async def load_session(id: str = Form(...), offset: int = Form(0)):
    data = agent.get_session_data(id)
    history = data.get("history", [])
    start = max(0, len(history) - offset - 10)
    return {"history": history[start : len(history) - offset], "has_more": start > 0}


@app.post("/chat")
async def chat(
    request: Request,
    message: str = Form(...),
    session_id: str = Form(...),
    model: str = Form(DEFAULT_MODEL),
    file: UploadFile = File(None),
):
    if not is_authorized(request):
        raise HTTPException(401)

    agent.stop_requested = False
    session_data = agent.get_session_data(session_id)
    file_ctx, images = await extract_file_context(file)

    user_msg = {"role": "user", "content": message}
    if images:
        user_msg["images"] = images
    session_data["history"].append(user_msg)

    async def generate():
        provider = get_provider_for_model(model)
        source = provider.source
        full_answer = ""

        try:
            logger.info("Chat con %s", model)
            yield "status:thinking:Kym está analizando...\n"

            if not model:
                raise RuntimeError(
                    f"No hay modelos disponibles. Inicia Ollama local, configura KYM_OLLAMA_HOST o define {NVIDIA_API_KEY_ENV}."
                )

            build_system_prompt = lambda extra: agent.build_prompt(session_data, extra)

            async for event in provider.stream_chat(
                model=model,
                history_messages=session_data["history"],
                build_system_prompt=build_system_prompt,
                extra_context=file_ctx,
                stop_requested=lambda: agent.stop_requested,
            ):
                if event.kind == "status":
                    yield f"status:{event.value}:{event.detail}\n"
                    continue

                full_answer += event.value
                yield strip_agent_commands(event.value)

            clean = strip_agent_commands(full_answer).strip()
            if agent.apply_saves(extract_saved_facts(full_answer)):
                yield "status:saving:Memorizando...\n"

            session_data["history"].append({"role": "assistant", "content": clean})
            agent.save_session_data(session_id, session_data)
            logger.info("Respuesta completada")

        except Exception as exc:
            exc = normalize_provider_error(source, exc, NVIDIA_API_KEY_ENV)
            logger.exception("Error durante el chat")
            yield f"\n⚠️ Error: {str(exc)}"

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/stop")
async def stop():
    agent.stop_requested = True
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    logger.info("Proveedor local apuntando a %s", local_provider.host)
    if local_provider.enabled and local_provider.get_client():
        local_provider.ensure_running()

    if os.getenv("IS_COLAB") == "true":
        print(f"\nTOKEN: {ACCESS_TOKEN}\n")
    else:
        from threading import Timer
        import webbrowser

        Timer(1.5, lambda: webbrowser.open("http://localhost:8000")).start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
