import asyncio
import re
from dataclasses import dataclass
from typing import Callable, Literal

COMMAND_PATTERN = re.compile(r"\[.*?\]")
SAVE_PATTERN = re.compile(r"\[SAVE:\s*(.*?)\s*\|\s*(.*?)\]")
SEARCH_PATTERN = re.compile(r"\[SEARCH:\s*(.*?)\]")
SYNC_STREAM_END = object()

SystemPromptBuilder = Callable[[str], str]
StopRequested = Callable[[], bool]


@dataclass(frozen=True)
class ProviderModel:
    id: str
    label: str
    source: str


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal["status", "text"]
    value: str
    detail: str = ""


def make_status_event(status: str, detail: str) -> StreamEvent:
    return StreamEvent(kind="status", value=status, detail=detail)


def make_text_event(text: str) -> StreamEvent:
    return StreamEvent(kind="text", value=text)


def strip_agent_commands(text: str) -> str:
    return COMMAND_PATTERN.sub("", text)


def extract_saved_facts(text: str):
    return SAVE_PATTERN.findall(text)


def build_api_messages(messages):
    return [{"role": item["role"], "content": item["content"]} for item in messages]


async def run_web_search(query: str):
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "La búsqueda web requiere la librería 'duckduckgo-search'."
        ) from exc

    def _search():
        return list(DDGS().text(keywords=query, max_results=3))

    return await asyncio.to_thread(_search)


def next_stream_chunk(stream_iterator):
    return next(stream_iterator, SYNC_STREAM_END)


async def iter_sync_stream(stream):
    stream_iterator = iter(stream)
    while True:
        chunk = await asyncio.to_thread(next_stream_chunk, stream_iterator)
        if chunk is SYNC_STREAM_END:
            break
        yield chunk


def normalize_provider_error(source: str, exc: Exception, api_key_env: str) -> Exception:
    detail = str(exc)
    status_code = getattr(exc, "status_code", None)

    if source == "api":
        if (
            status_code == 401
            or "Authentication failed" in detail
            or "Unauthorized" in detail
        ):
            return RuntimeError(
                f"Autenticación fallida con NVIDIA. Revisa la variable de entorno {api_key_env} y comprueba que la clave sea válida."
            )

    return exc
