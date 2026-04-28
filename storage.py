import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class AgentState:
    def __init__(self, model, memory_file="memory.json", sessions_dir="sessions", logger=None):
        self.model = model
        self.memory_file = Path(memory_file)
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.save_executor = ThreadPoolExecutor(max_workers=4)
        self.memory = self.load_memory()
        self.stop_requested = False

    def load_memory(self):
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as file_handle:
                    data = json.load(file_handle)
                    for key in ["sessions", "essential", "categories"]:
                        if key not in data:
                            data[key] = [] if key != "categories" else {}
                    return data
            except Exception as exc:
                self.logger.warning("No se pudo cargar la memoria: %s", exc)
        return {"essential": [], "categories": {}, "sessions": []}

    def save_memory(self):
        def _save():
            with open(self.memory_file, "w", encoding="utf-8") as file_handle:
                json.dump(self.memory, file_handle, indent=4, ensure_ascii=False)

        self.save_executor.submit(_save)

    def get_session_data(self, session_id):
        path = self.sessions_dir / f"{session_id}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as file_handle:
                    return json.load(file_handle)
            except Exception as exc:
                self.logger.warning("No se pudo cargar la sesion %s: %s", session_id, exc)
        return {"history": [], "active_categories": []}

    def save_session_data(self, session_id, data):
        def _save():
            path = self.sessions_dir / f"{session_id}.json"
            clean_history = []
            for message in data.get("history", []):
                msg = message.copy()
                if "images" in msg:
                    del msg["images"]
                clean_history.append(msg)

            with open(path, "w", encoding="utf-8") as file_handle:
                json.dump(
                    {
                        "history": clean_history,
                        "active_categories": data.get("active_categories", []),
                    },
                    file_handle,
                    indent=4,
                    ensure_ascii=False,
                )

        self.save_executor.submit(_save)

    def build_prompt(self, session_data, ext=""):
        essential = "\n".join(f"- {fact}" for fact in self.memory["essential"])
        categories = ", ".join(self.memory["categories"].keys())
        active = session_data.get("active_categories", [])
        active_str = "".join(
            f"\n[CAT {category.upper()}]: " + ", ".join(self.memory["categories"][category])
            for category in active
            if category in self.memory["categories"]
        )

        return (
            "ERES KYM, UN ASISTENTE ELITE DE IA. REGLAS CRITICAS:\n"
            "1. IDIOMA: Espanol profesional siempre.\n"
            "2. HERRAMIENTAS: Tienes acceso real a internet mediante comandos. "
            "Si necesitas informacion actual, usa [SEARCH: tema]. "
            "Si el usuario te pasa una URL o pregunta que pone en una web, usa [READ: https://...]. "
            "No inventes datos ni finjas haber leido o buscado si no ejecutaste una herramienta.\n"
            "3. EJECUCION: Cuando recibas resultados de herramienta, responde usando esos datos. "
            "Si el usuario pidio resumen, resume. Si no lo pidio, responde normal.\n"
            "4. AUTOCONTROL: Antes de responder, revisa si cumpliste toda la peticion, "
            "si usaste solo datos reales y si no hay enlaces o contenido inventado.\n\n"
            "EJEMPLOS:\n"
            "Usuario: 'Que tiempo hace en Madrid?'\n"
            "Kym: [SEARCH: clima en Madrid hoy]\n"
            "Usuario: 'Dime las noticias de hoy'\n"
            "Kym: [SEARCH: noticias destacadas hoy espana]\n"
            "Usuario: 'Que pone en esta web? https://pubmed.ncbi.nlm.nih.gov/32824977/'\n"
            "Kym: [READ: https://pubmed.ncbi.nlm.nih.gov/32824977/]\n\n"
            "5. FORMATO: Markdown claro y util.\n\n"
            f"MEMORIA ESENCIAL:\n{essential}\n"
            f"CATEGORIAS: {categories}\n{active_str}\n"
            "--- CONTEXTO ADICIONAL ---\n"
            f"{ext}\n"
            "RECUERDA: Si hace falta herramienta, emite solo el comando. Cuando el sistema te devuelva datos, responde con ellos."
        )

    def apply_saves(self, saves):
        if not saves:
            return False

        for category, fact in saves:
            category = category.strip().lower()
            fact = fact.strip()
            if category == "essential":
                self.memory["essential"].append(fact)
            else:
                if category not in self.memory["categories"]:
                    self.memory["categories"][category] = []
                self.memory["categories"][category].append(fact)

        self.save_memory()
        return True
