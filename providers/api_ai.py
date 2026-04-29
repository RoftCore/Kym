import os
from uuid import uuid4

from .base import (
    READ_PATTERN,
    SEARCH_PATTERN,
    ProviderModel,
    build_api_messages,
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


class OpenAICompatibleProvider:
    source = "api"

    def __init__(
        self,
        model_id: str,
        label: str,
        base_url: str,
        api_key: str = "",
        api_key_env: str = "",
        provider_id: str = "",
        source_label: str = "API",
    ):
        self.provider_id = provider_id or uuid4().hex[:10]
        self.model_id = model_id
        self.label = label
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.source_label = source_label
        self._api_key = api_key.strip()
        self._client = None

    def get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        if self.api_key_env:
            return os.getenv(self.api_key_env, "").strip()
        return ""

    def is_configured(self) -> bool:
        return bool(self.get_api_key() and self.base_url and self.model_id)

    def configure(
        self,
        *,
        model_id: str | None = None,
        label: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        source_label: str | None = None,
    ):
        if model_id is not None:
            self.model_id = model_id.strip()
        if label is not None:
            self.label = label.strip()
        if base_url is not None:
            self.base_url = base_url.strip()
        if api_key is not None:
            self._api_key = api_key.strip()
        if source_label is not None:
            self.source_label = source_label.strip() or "API"
        self._client = None

    def get_client(self):
        if self._client is not None:
            return self._client

        api_key = self.get_api_key()
        if not api_key:
            if self.api_key_env:
                raise RuntimeError(
                    f"Falta configurar la variable de entorno {self.api_key_env} para usar modelos API."
                )
            raise RuntimeError("Falta la API key para usar este proveedor.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "El modelo API requiere la libreria 'openai'. Instalala o elige un modelo local."
            ) from exc

        self._client = OpenAI(base_url=self.base_url, api_key=api_key)
        return self._client

    async def list_models(self):
        if not self.is_configured():
            return []
        return [ProviderModel(id=self.model_id, label=self.label, source=self.source)]

    async def stream_chat(
        self,
        model: str,
        history_messages,
        build_system_prompt,
        extra_context: str,
        stop_requested,
    ):
        client = self.get_client()
        full_messages = [
            {"role": "system", "content": build_system_prompt(extra_context)}
        ] + history_messages
        api_messages = build_api_messages(full_messages)
        compact = wants_summary(history_messages[-1]["content"])

        # Primera llamada para ver si quiere usar herramientas
        try:
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=150, # Suficiente para un comando
            )
            thought = response.choices[0].message.content or ""
        except Exception:
            thought = ""

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
            tool_result = render_search_response(search_results, query=query, compact=compact)

        if tool_result:
            # Añadir el pensamiento y el resultado al flujo
            api_messages.append({"role": "assistant", "content": thought})
            api_messages.append({"role": "user", "content": f"[SISTEMA: Datos obtenidos]\n{tool_result}\n\nResponde ahora al usuario basándote en estos datos."})
            yield make_status_event("writing", "Procesando información...")
        else:
            yield make_status_event("writing", "Redactando respuesta...")

        # Llamada final con streaming
        stream = client.chat.completions.create(
            model=model,
            messages=api_messages,
            stream=True,
        )
        async for chunk in iter_sync_stream(stream):
            if stop_requested():
                break
            text = chunk.choices[0].delta.content or ""
            if text:
                yield make_text_event(text)

    def to_config(self):
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "label": self.label,
            "base_url": self.base_url,
            "api_key": self._api_key,
            "source_label": self.source_label,
        }


class NvidiaAPIProvider(OpenAICompatibleProvider):
    def __init__(self, model_id: str, label: str, api_key_env: str, base_url: str):
        super().__init__(
            model_id=model_id,
            label=label,
            base_url=base_url,
            api_key_env=api_key_env,
            source_label="NVIDIA",
            provider_id="builtin-nvidia",
        )
