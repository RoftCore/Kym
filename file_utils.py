import base64
from io import BytesIO

from fastapi import UploadFile


def extract_pdf_text(raw: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Para analizar archivos PDF necesitas instalar la librería 'pypdf'."
        ) from exc

    pdf = PdfReader(BytesIO(raw))
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return f"--- CONTENIDO DEL PDF {filename} ---\n{text[:16000]}\n--- FIN DEL PDF ---"


async def extract_file_context(file: UploadFile | None):
    if not file:
        return "", []

    raw = await file.read()
    content_type = file.content_type or ""
    file_name = file.filename or "archivo"

    if content_type.startswith("image/"):
        image = base64.b64encode(raw).decode("utf-8")
        return f"[Imagen: {file_name}]", [image]

    if file_name.lower().endswith(".pdf"):
        return extract_pdf_text(raw, file_name), []

    text = raw.decode("utf-8", errors="ignore")[:16000]
    return f"--- CONTENIDO DEL ARCHIVO {file_name} ---\n{text}\n--- FIN DEL ARCHIVO ---", []
