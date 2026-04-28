import asyncio
import os
import socket
import subprocess
import time
from urllib.parse import urlparse

from .base import (
    READ_PATTERN,
    SEARCH_PATTERN,
    ProviderModel,
    format_search_results,
    make_status_event,
    make_text_event,
    normalize_search_query,
    render_read_response,
    render_search_response,
    run_url_read,
    run_web_search,
    wants_summary,
)


class LocalAIProvider:
    source = "local"

    def __init__(self, enabled: bool, host: str, logger):
        self.enabled = enabled
        self.host = self.normalize_host(host) or "http://127.0.0.1:11434"
        self.logger = logger
        self._client = None
        self._import_error = None

    def normalize_host(self, host: str | None):
        if host is None:
            return None
        clean_host = host.strip().rstrip("/")
        if not clean_host:
            return ""
        if "://" not in clean_host:
            if clean_host.startswith(("localhost", "127.0.0.1", "0.0.0.0")) or clean_host[:1].isdigit():
                clean_host = f"http://{clean_host}"
            else:
                clean_host = f"https://{clean_host}"
        return clean_host

    def configure(self, *, enabled: bool | None = None, host: str | None = None):
        if enabled is not None:
            self.enabled = enabled
        if host is not None:
            clean_host = self.normalize_host(host)
            if clean_host and clean_host != self.host:
                self.host = clean_host
                self._client = None

    def is_remote_host(self):
        parsed = urlparse(self.host)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False
        return hostname not in {"127.0.0.1", "localhost"}

    def is_ngrok_host(self):
        hostname = (urlparse(self.host).hostname or "").lower()
        return "ngrok" in hostname

    def can_autostart(self):
        return not self.is_remote_host()

    def build_client_headers(self):
        headers = {}
        if self.is_ngrok_host():
            headers["ngrok-skip-browser-warning"] = "true"
        return headers

    def get_client(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        if self._import_error is not None:
            return None
        try:
            import ollama
        except ImportError as exc:
            self._import_error = exc
            self.logger.warning(
                "La IA local no esta disponible porque falta la libreria 'ollama'."
            )
            return None
        self._client = ollama.AsyncClient(
            host=self.host,
            headers=self.build_client_headers(),
        )
        return self._client

    async def list_models(self):
        client = self.get_client()
        if not client:
            return []
        try:
            response = await client.list()
        except Exception as exc:
            self.logger.error("Error al listar modelos desde %s: %s", self.host, exc)
            return []

        models = []
        raw_models = []
        if isinstance(response, dict):
            raw_models = response.get("models", [])
        elif hasattr(response, "models"):
            raw_models = response.models

        for item in raw_models:
            name = None
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
            else:
                name = getattr(item, "name", None) or getattr(item, "model", None)
            if name:
                label = f"{name} (Colab)" if self.is_remote_host() else name
                models.append(ProviderModel(id=name, label=label, source=self.source))
        return models

    async def stream_chat(
        self,
        model: str,
        history_messages,
        build_system_prompt,
        extra_context: str,
        stop_requested,
    ):
        client = self.get_client()
        if not client:
            raise RuntimeError("Ollama no disponible.")

        full_messages = [
            {"role": "system", "content": build_system_prompt(extra_context)}
        ] + history_messages

        compact = wants_summary(history_messages[-1]["content"])
        response = await client.chat(model=model, messages=full_messages)
        thought = response["message"]["content"] or ""

        read = READ_PATTERN.search(thought)
        if read:
            url = read.group(1).strip()
            yield make_status_event("network", f"Leyendo contenido real de: '{url}'...")
            page = await run_url_read(url)
            answer = render_read_response(page, question=history_messages[-1]["content"])
            yield make_status_event("writing", "Redactando respuesta con datos actuales...")
            words = answer.split(" ")
            for i, word in enumerate(words):
                if stop_requested():
                    break
                yield make_text_event(word + (" " if i < len(words) - 1 else ""))
                await asyncio.sleep(0.005)
            return

        search = SEARCH_PATTERN.search(thought)
        if search:
            query = normalize_search_query(search.group(1))
            if not query:
                query = normalize_search_query(history_messages[-1]["content"])
            if not query:
                query = "noticias del dia"
            yield make_status_event("network", f"Buscando fuentes reales para: '{query}'...")
            search_results = await run_web_search(query)
            answer = render_search_response(search_results, query=query, compact=compact)
            yield make_status_event("writing", "Redactando respuesta con datos actuales...")
            words = answer.split(" ")
            for i, word in enumerate(words):
                if stop_requested():
                    break
                yield make_text_event(word + (" " if i < len(words) - 1 else ""))
                await asyncio.sleep(0.005)
            return

        yield make_status_event("writing", "Respondiendo...")
        final_text = thought.strip()
        if not final_text:
            final_text = format_search_results([], query=history_messages[-1]["content"], compact=compact)
        words = final_text.split(" ")
        for i, word in enumerate(words):
            if stop_requested():
                break
            yield make_text_event(word + (" " if i < len(words) - 1 else ""))
            await asyncio.sleep(0.005)

    def ensure_running(self):
        if self.is_remote_host():
            print(f"Ollama remoto configurado en {self.host}. No se inicia nada en local.")
            return True

        port = 11434
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                print("Ollama ya esta en ejecucion.")
                return True

        print("Ollama no detectado. Intentando iniciar servicio automaticamente...")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            for _ in range(10):
                time.sleep(1)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    if sock.connect_ex(("127.0.0.1", port)) == 0:
                        print("Servicio Ollama iniciado con exito.")
                        return True
            print("No se pudo iniciar Ollama automaticamente. Abrelo manualmente.")
        except Exception as exc:
            print(f"Error al intentar ejecutar Ollama: {exc}")
        return False
