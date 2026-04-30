import asyncio
import json
import os
import socket
import ssl
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .base import (
    READ_PATTERN,
    SEARCH_PATTERN,
    ProviderModel,
    iter_sync_stream,
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

        parsed = urlparse(clean_host)
        if not parsed.scheme or not parsed.netloc:
            return clean_host

        # Ollama expects the origin, not a concrete API path.
        # This lets the app accept either the base tunnel URL or a pasted /api/... URL.
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))

    def configure(self, *, enabled: bool | None = None, host: str | None = None):
        if enabled is not None:
            self.enabled = enabled
        if host is not None:
            clean_host = self.normalize_host(host)
            if clean_host and clean_host != self.host:
                self.host = clean_host

    def is_remote_host(self):
        parsed = urlparse(self.host)
        hostname = (parsed.hostname or "").lower()
        if not hostname:
            return False
        return hostname not in {"127.0.0.1", "localhost"}

    def is_ngrok_host(self):
        hostname = (urlparse(self.host).hostname or "").lower()
        return "ngrok" in hostname

    def is_bypass_required(self):
        hostname = (urlparse(self.host).hostname or "").lower()
        return self.is_ngrok_host() or "trycloudflare.com" in hostname

    def can_autostart(self):
        return not self.is_remote_host()

    def build_client_headers(self):
        headers = {
            "User-Agent": "curl/7.64.1",
        }
        if self.is_bypass_required():
            headers["ngrok-skip-browser-warning"] = "true"
        return headers

    def get_client(self):
        # Compatibilidad con el arranque existente en agent.py.
        # La implementación ya no usa un cliente persistente, pero el check
        # de arranque solo necesita una señal de "IA local disponible".
        return self if self.enabled else None

    def _ssl_context(self):
        if self.is_bypass_required():
            return ssl._create_unverified_context()
        return ssl.create_default_context()

    def _api_url(self, path: str):
        return f"{self.host}{path}"

    def _request_json(self, method: str, path: str, payload=None, timeout: int = 60):
        data = None
        headers = self.build_client_headers()
        headers["Accept"] = "application/json"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        request = Request(self._api_url(path), data=data, headers=headers, method=method)
        with urlopen(request, timeout=timeout, context=self._ssl_context()) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}

    def _stream_json_lines(self, method: str, path: str, payload=None, timeout: int = 60):
        data = None
        headers = self.build_client_headers()
        headers["Accept"] = "application/json"
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        request = Request(self._api_url(path), data=data, headers=headers, method=method)
        with urlopen(request, timeout=timeout, context=self._ssl_context()) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                yield json.loads(line)

    def _latest_user_images(self, history_messages):
        for message in reversed(history_messages or []):
            if (message.get("role") or "").lower() == "user" and message.get("images"):
                return message["images"]
        return []

    async def list_models(self):
        if not self.enabled:
            return []

        try:
            response = await asyncio.to_thread(self._request_json, "GET", "/api/tags")
        except Exception as exc:
            self.logger.error("Error al listar modelos desde %s: %s", self.host, exc)
            return []

        models = []
        for item in response.get("models", []):
            name = (item.get("name") or item.get("model") or "").strip()
            if not name:
                continue
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
        if not self.enabled:
            raise RuntimeError("Ollama local esta deshabilitado.")

        system_prompt = build_system_prompt(extra_context)
        messages = [{"role": "system", "content": system_prompt}] + history_messages
        images = self._latest_user_images(history_messages)

        def chat_once(stream: bool, message_list, image_list):
            payload = {
                "model": model,
                "messages": message_list,
                "stream": stream,
            }
            if image_list:
                payload["messages"][-1]["images"] = image_list
            if not stream:
                payload["options"] = {"num_predict": 150}
                return self._request_json("POST", "/api/chat", payload)
            return self._stream_json_lines("POST", "/api/chat", payload)

        try:
            response = await asyncio.to_thread(chat_once, False, messages, images)
            thought = ((response.get("message") or {}).get("content") or "").strip()
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise RuntimeError(f"No se pudo conectar con Ollama en {self.host}: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Error al consultar Ollama en {self.host}: {exc}") from exc

        read = READ_PATTERN.search(thought)
        search = SEARCH_PATTERN.search(thought)

        tool_result = None
        if read:
            url = read.group(1).strip()
            yield make_status_event("network", f"Leyendo contenido real de: '{url}'...")
            page = await run_url_read(url)
            tool_result = render_read_response(page, question=history_messages[-1]["content"])
        elif search:
            query = normalize_search_query(search.group(1))
            if not query:
                query = normalize_search_query(history_messages[-1]["content"])
            if not query:
                query = "noticias del dia"
            yield make_status_event("network", f"Buscando en internet: '{query}'...")
            search_results = await run_web_search(query)
            compact = wants_summary(history_messages[-1]["content"])
            tool_result = render_search_response(search_results, query=query, compact=compact)

        if tool_result:
            messages.append({"role": "assistant", "content": thought})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[SISTEMA: Datos obtenidos]\n"
                        f"{tool_result}\n\n"
                        "Responde ahora al usuario basandote en estos datos."
                    ),
                }
            )
            yield make_status_event("writing", "Procesando informacion...")
        else:
            yield make_status_event("writing", "Respondiendo...")

        stream = await asyncio.to_thread(chat_once, True, messages, self._latest_user_images(messages[1:]))
        async for chunk in iter_sync_stream(stream):
            if stop_requested():
                break
            token = ((chunk.get("message") or {}).get("content") or "").strip()
            if token:
                yield make_text_event(token)

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
