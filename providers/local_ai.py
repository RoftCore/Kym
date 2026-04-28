import os
import socket
import subprocess
import time
import asyncio
import re
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
                label = name
                if self.is_remote_host():
                    label = f"{name} (Colab)"
                models.append(ProviderModel(id=name, label=label, source=self.source))
        return models

    def _clean_query(self, text: str):
        """Limpia la query manteniendo las palabras clave importantes"""
        import re
        
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar frases comunes al inicio
        text = re.sub(r'^(dame|dime|busca|quiero|necesito|por favor)\s+', '', text)
        
        # Eliminar palabras al final
        text = re.sub(r'\s+(por favor|gracias)$', '', text)
        
        # Eliminar artículos y preposiciones (pero mantener palabras clave)
        text = re.sub(r'\b(de|la|las|los|y|el|lo|un|una|unos|unas|del|al)\b', ' ', text)
        
        # Limpiar espacios múltiples
        text = ' '.join(text.split())
        
        # Si la query es solo un número o muy corta, añadir contexto
        if re.match(r'^\d+$', text):
            text = f"noticias {text}"
        
        # Si la query es "dia" o "día", añadir "noticias"
        if text in ["dia", "día", "hoy"]:
            text = f"noticias {text}"
        
        # Si la query sigue siendo muy corta, usar default
        if len(text) < 5:
            return "noticias actualidad"
        
        return text

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

        self.logger.info("=== PRIMERA LLAMADA AL MODELO ===")
        response = await client.chat(model=model, messages=full_messages)
        thought = response["message"]["content"]
        self.logger.info(f"Pensamiento recibido (longitud: {len(thought)} chars)")

        search = SEARCH_PATTERN.search(thought)
        intent_keywords = ["enlace", "link", "fuente", "noticia", "buscar", "search", "noticias"]
        has_intent = any(kw in thought.lower() for kw in intent_keywords)
        
        self.logger.info(f"¿Búsqueda detectada? search={search is not None}, has_intent={has_intent}")
        
        if search or has_intent:
            raw_query = search.group(1) if search else history_messages[-1]["content"]
            query = self._clean_query(raw_query)
            self.logger.info(f"Query limpia: '{query}'")
            
            yield make_status_event("network", f"Buscando en tiempo real: '{query}'...")
            
            try:
                search_results = await run_web_search(query)
                self.logger.info(f"Resultados obtenidos: {len(search_results)}")
                
                if search_results:
                    formatted_results = "\n".join(search_results[:5])  # Limitar a 5 resultados
                    self.logger.info(f"Resultados formateados (primeros 200 chars): {formatted_results[:200]}...")
                    
                    full_messages.append({"role": "assistant", "content": thought})
                    full_messages.append({
                        "role": "user", 
                        "content": (
                            f"RESULTADOS DE INTERNET (con links reales):\n{formatted_results}\n\n"
                            "INSTRUCCIONES IMPORTANTES:\n"
                            "1. Los links (🔗 o 📎) son URLs reales y funcionales\n"
                            "2. DEBES incluir los links originales en tu respuesta\n"
                            "3. Si el usuario pide links, proporciona los que están en los resultados\n"
                            "4. No digas que no hay links si aparecen en los resultados\n"
                            "5. Responde en español de forma clara y concisa"
                        )
                    })
                    
                    self.logger.info("=== SEGUNDA LLAMADA AL MODELO (STREAM) ===")
                    yield make_status_event("writing", "Procesando información y redactando...")
                    
                    stream = await client.chat(model=model, messages=full_messages, stream=True)
                    
                    chunk_count = 0
                    async for chunk in stream:
                        if stop_requested():
                            self.logger.info("Stop requested, interrumpiendo stream")
                            break
                        
                        chunk_count += 1
                        text = chunk["message"]["content"]
                        
                        if text:
                            self.logger.debug(f"Chunk {chunk_count}: '{text[:50]}'")
                            yield make_text_event(text)
                    
                    self.logger.info(f"Stream finalizado. Total chunks: {chunk_count}")
                    
                    if chunk_count == 0:
                        self.logger.error("¡No se recibieron chunks del stream!")
                        yield make_text_event("Lo siento, no pude generar una respuesta con los resultados de búsqueda.")
                    
                    return 
                else:
                    self.logger.warning("No se obtuvieron resultados de búsqueda")
                    yield make_text_event("No encontré resultados relevantes para tu búsqueda. ¿Podrías reformular la pregunta?")
                    
            except Exception as e:
                self.logger.error(f"Error en búsqueda: {e}", exc_info=True)
                yield make_text_event(f"Error al buscar: {str(e)}")
        else:
            self.logger.info("No se detectó intención de búsqueda, usando respuesta directa")
            yield make_status_event("writing", "Respondiendo...")
            yield make_text_event(thought)

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
