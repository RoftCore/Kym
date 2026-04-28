import os
import socket
import subprocess
import time
from urllib.parse import urlparse

from .base import (
    ProviderModel,
    SEARCH_PATTERN,
    make_status_event,
    make_text_event,
    run_web_search,
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
                "La IA local no está disponible porque falta la librería 'ollama'."
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
        for item in response.get("models", []):
            name = item.get("name")
            if name:
                label = name
                if self.is_remote_host():
                    label = f"{name} (Colab)"
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
            if not self.enabled:
                raise RuntimeError(
                    "La IA local está deshabilitada en este arranque. Elige un modelo API o reinicia sin --no-local."
                )
            raise RuntimeError(
                "No se pudo cargar la librería 'ollama'. Instálala solo si vas a usar IA local."
            )

        full_messages = [
            {"role": "system", "content": build_system_prompt(extra_context)}
        ] + history_messages

        response = await client.chat(model=model, messages=full_messages)
        thought = response["message"]["content"]

        search = SEARCH_PATTERN.search(thought)
        if search:
            query = search.group(1)
            yield make_status_event("network", f"Buscando en internet: {query}...")
            search_results = await run_web_search(query)
            extra_context += f"\n[Resultados Web]: {search_results}"
            full_messages[0]["content"] = build_system_prompt(extra_context)

        yield make_status_event("writing", "Redactando respuesta...")
        stream = await client.chat(model=model, messages=full_messages, stream=True)
        async for chunk in stream:
            if stop_requested():
                break
            text = chunk["message"]["content"]
            if text:
                yield make_text_event(text)

    def ensure_running(self):
        """Comprueba si Ollama está activo y, si no, intenta iniciarlo."""
        if self.is_remote_host():
            print(f"Ollama remoto configurado en {self.host}. No se inicia nada en local.")
            return True

        port = 11434
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                print("Ollama ya está en ejecución.")
                return True

        print("Ollama no detectado. Intentando iniciar servicio automáticamente...")
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
                        print("Servicio Ollama iniciado con éxito.")
                        return True

            print("No se pudo iniciar Ollama automáticamente. Ábrelo manualmente.")
        except Exception as exc:
            print(f"Error al intentar ejecutar Ollama: {exc}")
        return False
