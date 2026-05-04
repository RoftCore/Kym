import requests
import json
import os
import sys
import subprocess
import urllib.parse
from html.parser import HTMLParser

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"
IA_LOCAL_DIR = "./IA-Local"
FILES = ["guide.md", "memory.md", "skills.md"]

# ==============================================================================
# HERRAMIENTAS
# ==============================================================================

def ejecutar_comando(comando):
    """Ejecuta un comando del sistema y devuelve el resultado."""
    print(f"\n[EJECUTANDO] {comando}")
    try:
        resultado = subprocess.run(comando, shell=True, capture_output=True, text=True, timeout=30)
        if resultado.returncode == 0:
            output = resultado.stdout.strip()
            if not output:
                output = "[Comando ejecutado correctamente, sin salida de texto]"
            return output
        else:
            return f"[ERROR] {resultado.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "[ERROR] El comando tardó demasiado y fue cancelado."
    except Exception as e:
        return f"[ERROR] {e}"

def buscar_en_internet(query):
    """Busca en DuckDuckGo y devuelve los primeros resultados."""
    print(f"\n[BUSCANDO] {query}")
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=10)
        
        if res.status_code != 200:
            return f"[ERROR] Error al buscar: HTTP {res.status_code}"

        class ExtractResults(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.capture = False
                self.current = {}
                self.data_tag = None

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if tag == 'a' and 'result__a' in attrs_dict.get('class', ''):
                    self.capture = True
                    self.current = {'title': '', 'link': '', 'snippet': ''}
                if self.capture and tag == 'a':
                    self.current['link'] = attrs_dict.get('href', '')
                if tag == 'a' and 'result__snippet' in attrs_dict.get('class', ''):
                    self.data_tag = 'snippet'

            def handle_data(self, data):
                if self.capture:
                    if self.data_tag == 'snippet':
                        self.current['snippet'] += data
                    else:
                        self.current['title'] += data

            def handle_endtag(self, tag):
                if self.capture and tag == 'a' and self.current['title']:
                    self.results.append(dict(self.current))
                    self.capture = False
                    self.current = {}
                self.data_tag = None

        parser = ExtractResults()
        parser.feed(res.text)
        
        if not parser.results:
            return "[AVISO] No se encontraron resultados."

        formatted = []
        for i, r in enumerate(parser.results[:3], 1):
            link_limpio = r.get('link', '').replace('//', 'https://') if r.get('link', '').startswith('//') else r.get('link', '')
            formatted.append(f"{i}. {r['title'].strip()}\n   {r['snippet'].strip()}\n   {link_limpio}")
        
        return "\n\n".join(formatted)

    except Exception as e:
        return f"[ERROR] Fallo en la búsqueda: {e}"

# ==============================================================================
# LÓGICA PRINCIPAL
# ==============================================================================

def check_ollama():
    try:
        test = requests.get("http://localhost:11434/api/tags", timeout=5)
        return test.status_code == 200
    except:
        return False

def load_context():
    """Carga guía, memoria y skills desde archivos. Crea el directorio y archivos si no existen."""
    if not os.path.exists(IA_LOCAL_DIR):
        os.makedirs(IA_LOCAL_DIR, exist_ok=True)
        print(f"[OK] Directorio creado: {IA_LOCAL_DIR}")

    context = ""
    for f in FILES:
        filepath = os.path.join(IA_LOCAL_DIR, f)
        if not os.path.exists(filepath):
            try:
                with open(filepath, "w", encoding="utf-8") as file:
                    file.write(f"# {f}\n")
                print(f"[AVISO] {f} no encontrado. Creado archivo básico.")
            except Exception as e:
                print(f"[ERROR] No se pudo crear {f}: {e}")
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    context += content + "\n\n"
                    print(f"[OK] {f} cargado")
        except Exception as e:
            print(f"[ERROR] No se pudo leer {f}: {e}")
    return context

def update_memory(new_entry):
    filepath = os.path.join(IA_LOCAL_DIR, "memory.md")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n{new_entry}")
    print("[OK] Memoria actualizada")

def parse_tool_calls(text):
    """
    Busca [BUSCAR: ...] o [CMD: ...] en la respuesta del modelo.
    Devuelve (tipo, argumento) o (None, None).
    """
    import re
    # Buscar [BUSCAR: query]
    match = re.search(r'\[BUSCAR:\s*(.+?)\]', text, re.IGNORECASE)
    if match:
        return "buscar", match.group(1).strip()
    # Buscar [CMD: comando]
    match = re.search(r'\[CMD:\s*(.+?)\]', text, re.IGNORECASE)
    if match:
        return "cmd", match.group(1).strip()
    return None, None

def chat(prompt, context=""):
    """Envía prompt a Ollama con streaming y gestiona herramientas automáticamente."""
    herramientas_prompt = """
[DISPONIBLE]
Puedes usar estas herramientas escribiendo exactamente:
- [BUSCAR: texto a buscar] para buscar en internet
- [CMD: comando] para ejecutar un comando del sistema
Úsalas solo cuando sea necesario. El usuario no las ve, se ejecutan automáticamente.

**IMPORTANTE**: Si el usuario te pregunta algo que no sabes (fechas, clima, noticias, información actual), USA [BUSCAR:] o [CMD:] inmediatamente. NO respondas con información de tu guía si no es relevante para la pregunta.
No uses [MEMORIA] ni [WRITE], usa [BUSAR: texto a buscar].
[/DISPONIBLE]
"""
    full_prompt = f"{context}\n{herramientas_prompt}\n\n---\nUsuario: {prompt}\nAsistente:"
    
    try:
        response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": full_prompt, "stream": True}, stream=True, timeout=120)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    full_response += data["response"]
        
        # --- Procesar herramientas ---
        tipo, args = parse_tool_calls(full_response)
        if tipo == "buscar":
            resultado = buscar_en_internet(args)
            # Llamada recursiva con el resultado
            prompt_ampliado = f"[RESULTADO BUSQUEDA para '{args}':\n{resultado}]\n\nBasándote en esto, responde al usuario."
            return chat(prompt_ampliado, context)
        elif tipo == "cmd":
            resultado = ejecutar_comando(args)
            prompt_ampliado = f"[RESULTADO COMANDO '{args}':\n{resultado}]\n\nBasándote en esto, responde al usuario."
            return chat(prompt_ampliado, context)
        
        return full_response

    except Exception as e:
        return f"[ERROR] {e}"

def main():
    print("=" * 50)
    print(f"Kym V2 — Asistente Autónomo con {MODEL}")
    print("=" * 50)

    if not check_ollama():
        print("[ERROR] Ollama no está corriendo. Ejecuta 'ollama serve' en otra terminal.")
        sys.exit(1)
    print("[OK] Ollama detectado")

    context = load_context()
    print("\nLa IA decide cuándo ejecutar [CMD] o [BUSCAR].\n")

    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            break
        
        respuesta = chat(user_input, context)
        print(f"Asistente: {respuesta}\n")

if __name__ == "__main__":
    main()