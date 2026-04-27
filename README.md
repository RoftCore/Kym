# 🤖 Elite Agent: Asistente Inteligente Multisesión

Elite Agent es un asistente de IA avanzado construido sobre **Ollama** y **FastAPI**. Está diseñado para ser eficiente, consciente de su propia memoria y capaz de navegar por internet para proporcionar respuestas precisas y actualizadas.

## 🚀 Características Principales

- **🧠 Memoria Arquitectónica Modular**: Separa hechos esenciales de contextos temáticos (Negocio, Salud, Proyectos, etc.). El agente activa estos contextos solo cuando es necesario.
- **🌐 Acceso a Internet Real**: Capacidad para buscar en DuckDuckGo y leer el contenido completo de sitios web mediante web scraping.
- **📂 Gestión de Sesiones**: Panel lateral para manejar múltiples conversaciones independientes. Las sesiones se guardan automáticamente.
- **👻 Modo Efímero**: Opción para crear chats temporales que no se guardan en el historial permanente.
- **🔄 Razonamiento Recursivo**: El agente puede realizar múltiples pasos de "pensamiento" (buscar -> leer -> analizar) antes de responder.
- **🎨 Interfaz Web Moderna**: Diseño oscuro profesional con soporte completo para **Markdown**, resaltado de sintaxis de código y tablas.
- **⚡ Precarga Automática (Warm-up)**: Carga el modelo en RAM/VRAM al iniciar para que la primera respuesta sea instantánea.

## 🛠️ Instalación

1. **Requisitos Previos**:
   - Tener instalado [Ollama](https://ollama.com/).
   - Descargar el modelo: `ollama pull llama3.1`.
   - Python 3.10 o superior.

2. **Instalar Dependencias**:
   ```bash
   pip install fastapi uvicorn ollama duckduckgo-search beautifulsoup4 requests
   ```

3. **Ejecutar**:
   ```bash
   python agent.py
   ```
   *El navegador se abrirá automáticamente en `http://localhost:8000`.*

## 📂 Estructura del Proyecto

- `agent.py`: Servidor backend y lógica del agente.
- `static/`: Archivos de la interfaz (HTML, CSS, JS).
- `sessions/`: Almacén de historiales de conversación (JSON).
- `memory.json`: Base de datos de memoria a largo plazo (Esencial y Categorías).
- `skills/`: Instrucciones personalizadas para habilidades específicas.

## 📝 Comandos Internos del Agente

El agente gestiona estas acciones de forma autónoma, pero puedes influir en ellas:
- `[SEARCH: consulta]`: El agente decide buscar en la red.
- `[READ: url]`: El agente decide profundizar en una web.
- `[SAVE: categoria | dato]`: El agente decide guardar algo en tu memoria core o temática.
- `[LOAD: categoria]`: El agente recupera información de un grupo específico.

## 🛡️ Licencia
Este proyecto es de uso libre para experimentación y desarrollo personal.
