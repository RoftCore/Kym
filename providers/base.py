import asyncio
import re
import warnings
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Callable, Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")

COMMAND_PATTERN = re.compile(r"\[(SEARCH|SAVE|LOAD|READ):.*?\]", re.IGNORECASE)
SAVE_PATTERN = re.compile(r"\[SAVE:\s*(.*?)\s*\|\s*(.*?)\]", re.IGNORECASE)
SEARCH_PATTERN = re.compile(r"\[SEARCH:\s*(.*?)\]", re.IGNORECASE)
READ_PATTERN = re.compile(r"\[READ:\s*(https?://[^\]]+)\]", re.IGNORECASE)
EXAMPLE_LINK_PATTERN = re.compile(r"example\.com", re.IGNORECASE)
LIST_ITEM_PATTERN = re.compile(r"(?m)^\s*(?:\d+[\.)]|[-*])\s+")
SUMMARY_HINT_PATTERN = re.compile(
    r"\b(resumid[oa]s?|resumen|resumir|breve|brevemente)\b",
    re.IGNORECASE,
)
SYNC_STREAM_END = object()

SystemPromptBuilder = Callable[[str], str]
StopRequested = Callable[[], bool]


class SimpleHTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.in_style = False
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        lower = tag.lower()
        if lower == "script":
            self.in_script = True
        elif lower == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        lower = tag.lower()
        if lower == "script":
            self.in_script = False
        elif lower == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_script or self.in_style:
            return
        text = " ".join(data.split())
        if text:
            self.text_parts.append(text)


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


@dataclass(frozen=True)
class PageReadResult:
    url: str
    title: str
    summary: str
    content: str


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
    for candidate in (
        base,
        f"{base} noticias" if base else "",
        f"{base} hoy" if base else "",
        f"ultimas noticias {base}" if base else "",
        f"{base} actualidad" if base else "",
    ):
        candidate = normalize_search_query(candidate)
        if candidate and candidate not in variants and len(candidate) > 3:
            variants.append(candidate)
    if not variants:
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
    return cleaned.strip(" -:|")


def summarize_search_text(text: str, limit: int = 170) -> str:
    cleaned = clean_search_text(text)
    cleaned = re.split(r"\s*[•·|-]\s*", cleaned, maxsplit=1)[0].strip()
    cleaned = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def parse_search_results(results, limit: int = 3):
    items = []
    for raw in (results or [])[:limit]:
        title = ""
        link = ""
        summary = ""
        for line in str(raw).splitlines():
            clean_line = line.strip()
            upper_line = clean_line.upper()
            if upper_line.startswith("FUENTE:"):
                title = clean_line.split(":", 1)[1].strip()
            elif upper_line.startswith("LINK:"):
                link = clean_line.split(":", 1)[1].strip()
            elif upper_line.startswith("RESUMEN:"):
                summary = clean_line.split(":", 1)[1].strip()
        if title or link or summary:
            items.append(
                SearchResultItem(
                    title=clean_search_text(title) or "Sin titular",
                    link=link,
                    summary=summarize_search_text(summary, limit=220),
                )
            )
    return items


def _extract_meta(html: str, attr_name: str, attr_value: str):
    patterns = [
        rf'<meta[^>]+{attr_name}=["\']{re.escape(attr_value)}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+{attr_name}=["\']{re.escape(attr_value)}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return unescape(match.group(1)).strip()
    return ""


def _extract_title(html: str):
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return unescape(" ".join(match.group(1).split())).strip()


def _extract_pubmed_abstract(html: str):
    matches = re.findall(
        r'<div[^>]+class=["\'][^"\']*abstract-content[^"\']*["\'][^>]*>(.*?)</div>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    parts = []
    for chunk in matches:
        text = re.sub(r"<[^>]+>", " ", chunk)
        text = " ".join(unescape(text).split())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _extract_generic_text(html: str):
    parser = SimpleHTMLTextExtractor()
    parser.feed(html)
    text = " ".join(parser.text_parts)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _truncate(text: str, limit: int):
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


async def run_web_search(query: str):
    try:
        warnings.filterwarnings("ignore", message="This package .* has been renamed to .*")
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "La busqueda web requiere la libreria 'ddgs' (pip install ddgs)"
        ) from exc

    def _search():
        with DDGS() as ddgs:
            for candidate in build_search_variants(query):
                try:
                    results = list(ddgs.text(query=candidate, max_results=5))
                except TypeError:
                    results = list(ddgs.text(candidate, max_results=5))
                except Exception:
                    continue

                formatted = []
                for result in results:
                    title = (result.get("title") or "").strip()
                    link = (result.get("href") or "").strip()
                    body = (result.get("body") or "").strip()
                    if not (title or link or body):
                        continue
                    formatted.append(
                        f"FUENTE: {title}\nLINK: {link}\nRESUMEN: {body}\n---"
                    )
                if formatted:
                    return formatted
            return []

    return await asyncio.to_thread(_search)


async def run_url_read(url: str):
    def _read():
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; KymBot/1.0; +https://localhost)",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            },
        )
        with urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            charset_match = re.search(r"charset=([^\s;]+)", content_type, flags=re.IGNORECASE)
            charset = charset_match.group(1) if charset_match else "utf-8"
            raw = response.read()
        html = raw.decode(charset, errors="replace")

        title = (
            _extract_meta(html, "name", "citation_title")
            or _extract_meta(html, "property", "og:title")
            or _extract_title(html)
        )
        summary = (
            _extract_meta(html, "name", "description")
            or _extract_meta(html, "property", "og:description")
            or ""
        )

        parsed = urlparse(url)
        if "pubmed.ncbi.nlm.nih.gov" in parsed.netloc.lower():
            content = _extract_pubmed_abstract(html)
            if not summary:
                summary = content
        else:
            content = _extract_generic_text(html)

        content = _truncate(content, 4000)
        summary = _truncate(" ".join((summary or "").split()), 800)

        return PageReadResult(
            url=url,
            title=title or parsed.netloc or url,
            summary=summary,
            content=content,
        )

    return await asyncio.to_thread(_read)


def render_search_response(results, query: str = "", compact: bool = False, limit: int = 3) -> str:
    items = parse_search_results(results, limit=limit)
    if not items:
        if query:
            return f"No encontre resultados utiles para: {query}."
        return "No encontre resultados utiles."

    header = "Aqui tienes 3 noticias resumidas" if compact else "Aqui tienes resultados reales"
    if query:
        header += f" para {query}"
    header += ":"

    lines = [header]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. **{item.title}**")
        lines.append(f"Resumen: {item.summary or 'Sin resumen disponible.'}")
        if item.link and not compact:
            lines.append(f"Fuente: {item.link}")
    return "\n".join(lines)


def render_read_response(page: PageReadResult, question: str = "") -> str:
    lines = [f"Contenido leido de: {page.url}"]
    if page.title:
        lines.append(f"Titulo: {page.title}")
    if page.summary:
        lines.append(f"Resumen de la pagina: {page.summary}")
    if page.content:
        lines.append("Texto extraido:")
        lines.append(page.content)
    if question:
        lines.append(f"Consulta original: {question}")
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
