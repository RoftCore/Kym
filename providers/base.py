import asyncio
import re
import warnings
from dataclasses import dataclass
from typing import Callable, Literal

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

COMMAND_PATTERN = re.compile(r"\[.*?\]")
SAVE_PATTERN = re.compile(r"\[SAVE:\s*(.*?)\s*\|\s*(.*?)\]")
SEARCH_PATTERN = re.compile(r"\[SEARCH:\s*(.*?)\]")
EXAMPLE_LINK_PATTERN = re.compile(r"example\.com", re.IGNORECASE)
LIST_ITEM_PATTERN = re.compile(r"(?m)^\s*(?:\d+[\.)]|[-*])\s+")
SUMMARY_HINT_PATTERN = re.compile(
    r"\b(resumid[oa]s?|resumen|resumir|breve|brevemente)\b",
    re.IGNORECASE,
)
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


@dataclass(frozen=True)
class SearchResultItem:
    title: str
    link: str
    summary: str


def make_status_event(status: str, detail: str) -> StreamEvent:
    return StreamEvent(kind="status", value=status, detail=detail)


def make_text_event(text: str) -> StreamEvent:
    return StreamEvent(kind="text", value=text)


def strip_agent_commands(text: str) -> str:
    return re.sub(r"\[(SEARCH|SAVE|LOAD):.*?\]", "", text, flags=re.IGNORECASE)


def extract_saved_facts(text: str):
    return SAVE_PATTERN.findall(text)


def build_api_messages(messages):
    return [{"role": item["role"], "content": item["content"]} for item in messages]


def wants_summary(text: str) -> bool:
    return bool(SUMMARY_HINT_PATTERN.search(text or ""))


def normalize_search_query(query: str) -> str:
    cleaned = " ".join((query or "").split())
    cleaned = cleaned.strip(" .,:;!?\"'`()[]{}")
    cleaned = re.sub(
        r"\b(en el|en la|del|de la|de los|de las|por favor)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return " ".join(cleaned.split())


def build_search_variants(query: str):
    base = normalize_search_query(query)
    variants = []
    
    # Variantes más efectivas para noticias
    for candidate in (
        base,
        f"{base} noticias" if base else "",
        f"{base} hoy" if base else "",
        f"ultimas noticias {base}" if base else "",
        f"{base} actualidad" if base else "",
        base,  # repetir base al final
    ):
        candidate = normalize_search_query(candidate)
        if candidate and candidate not in variants and len(candidate) > 3:
            variants.append(candidate)
    
    # Si no hay variantes o son muy pocas, añadir términos generales
    if not variants or len(variants) < 2:
        variants.append("noticias actualidad")
    
    return variants


def count_list_items(text: str) -> int:
    return len(LIST_ITEM_PATTERN.findall(text or ""))


def validate_search_answer(text: str, compact: bool = False, required_items: int = 3):
    issues = []
    content = text or ""
    lowered = content.lower()

    if EXAMPLE_LINK_PATTERN.search(content):
        issues.append("contiene example.com")

    if compact:
        item_count = count_list_items(content)
        if item_count < required_items:
            issues.append(f"solo tiene {item_count} elementos y se pedian {required_items}")
        if "resumen" not in lowered:
            issues.append("no incluye resumenes")

    return issues


def clean_search_text(text: str) -> str:
    cleaned = " ".join((text or "").split())
    cleaned = re.sub(
        r"\s*-\s*\d+\s+(minute|hour|day|week|month|year)s?\s+ago\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^ultimas noticias:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^noticias de ultima hora\b[:\-\s]*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^noticias,\s*la ultima hora de espana y el mundo\b[:\-\s]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    for prefix in (
        "Consulta las ultimas noticias de actualidad al minuto y la ultima hora de Espana, Mexico, America y el mundo en ",
        "Noticias de ultima hora en Espana y el mundo. Toda la actualidad y las ultimas noticias nacionales e internacionales aqui, en ",
        "Listado de las ultimas noticias publicadas en ",
    ):
        cleaned = cleaned.replace(prefix, "")
    cleaned = re.sub(
        r"\s+[–-]\s+(EL PAIS|EL MUNDO|RTVE\.es?)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" -–:|")


def summarize_search_text(text: str, limit: int = 170) -> str:
    cleaned = clean_search_text(text)
    cleaned = re.split(r"\s*[•·|-]\s*", cleaned, maxsplit=1)[0].strip()
    cleaned = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def parse_search_results(results, limit: int = 3):
    """Parsea los resultados de búsqueda correctamente"""
    items = []
    
    for raw in (results or [])[:limit]:
        # Si raw es string, parsear líneas
        if isinstance(raw, str):
            title = ""
            link = ""
            summary = ""
            
            for line in raw.split('\n'):
                line = line.strip()
                if line.startswith("FUENTE:"):
                    title = line.replace("FUENTE:", "").strip()
                elif line.startswith("LINK:"):
                    link = line.replace("LINK:", "").strip()
                elif line.startswith("RESUMEN:"):
                    summary = line.replace("RESUMEN:", "").strip()
            
            # Si no se encontró título pero hay link, usar el dominio
            if not title and link:
                from urllib.parse import urlparse
                domain = urlparse(link).netloc
                title = domain.replace('www.', '')
            
            if summary or link:
                items.append(SearchResultItem(
                    title=title or "Noticia",
                    link=link,
                    summary=summary[:300]  # Limitar longitud
                ))
        else:
            # Si ya es un objeto SearchResultItem
            items.append(raw)
    
    return items


async def run_web_search(query: str):
    try:
        warnings.filterwarnings(
            "ignore",
            message="This package .* has been renamed to .*",
        )
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "La busqueda web requiere la libreria 'ddgs' (pip install ddgs)"
        ) from exc

    def _search():
        with DDGS() as ddgs:
            for candidate in build_search_variants(query):
                try:
                    # CORRECCIÓN: Usar 'query=' como argumento nombrado
                    results = list(ddgs.text(query=candidate, max_results=5))
                    
                    if not results:
                        continue
                    
                    formatted = []
                    for result in results:
                        title = result.get("title", "").strip()
                        link = result.get("href", "").strip()
                        body = result.get("body", "").strip()
                        
                        if body or link:
                            formatted.append(
                                f"FUENTE: {title}\nLINK: {link}\nRESUMEN: {body}\n---"
                            )
                    
                    if formatted:
                        return formatted
                        
                except TypeError as te:
                    # Si falla con argumento nombrado, intentar posicional
                    try:
                        results = list(ddgs.text(candidate, max_results=5))
                        formatted = []
                        for result in results:
                            title = result.get("title", "").strip()
                            link = result.get("href", "").strip()
                            body = result.get("body", "").strip()
                            if body or link:
                                formatted.append(
                                    f"FUENTE: {title}\nLINK: {link}\nRESUMEN: {body}\n---"
                                )
                        if formatted:
                            return formatted
                    except Exception as e2:
                        print(f"Error con argumento posicional: {e2}")
                        
                except Exception as e:
                    print(f"Error en búsqueda con '{candidate}': {e}")
                    continue
            
            return []

    return await asyncio.to_thread(_search)


def render_search_response(results, query: str = "", compact: bool = False, limit: int = 3) -> str:
    items = parse_search_results(results, limit=limit)
    if not items:
        if query:
            return f"No encontre resultados utiles para: {query}."
        return "No encontre resultados utiles."

    if compact:
        header = "📰 **RESUMEN DE NOTICIAS**\n"
        lines = [header]
        for index, item in enumerate(items, start=1):
            title = item.title or "Sin titular"
            summary = summarize_search_text(item.summary, limit=150)
            if not summary:
                summary = "Sin resumen disponible."
            lines.append(f"\n**{index}. {title}**")
            lines.append(f"📝 {summary}")
            if item.link:
                lines.append(f"🔗 {item.link}")
        return "\n".join(lines)
    else:
        header = f"🔍 Resultados para: {query}\n" if query else "🔍 Resultados encontrados:\n"
        lines = [header]
        for index, item in enumerate(items, start=1):
            lines.append(f"\n{index}. **{item.title}**")
            if item.summary:
                lines.append(f"   {item.summary}")
            if item.link:
                lines.append(f"   📎 {item.link}")
        return "\n".join(lines)


def format_search_results(results, query: str = "", limit: int = 3, compact: bool = False) -> str:
    return render_search_response(results, query=query, compact=compact, limit=limit)


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
                f"Autenticacion fallida con NVIDIA. Revisa la variable de entorno {api_key_env} y comprueba que la clave sea valida."
            )

    return exc
