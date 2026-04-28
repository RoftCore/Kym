# Kym

Kym es un asistente web construido con FastAPI. Puede funcionar con IA local via Ollama o con modelos por API, y guarda sesiones y memoria en archivos JSON.

## Caracteristicas

- Interfaz web con historial de sesiones.
- Modo efimero para chats que no se guardan.
- Soporte para archivos e imagenes.
- Memoria persistente por categorias.
- Opcion de usar IA local o modelos por API.

## Instalacion

Requisitos:

- Python 3.10 o superior.
- Ollama solo si vas a usar IA local.

Instalacion base del servidor:

```bash
pip install -r requirements.txt
```

Instalacion con IA local:

```bash
pip install -r requirements-local.txt
```

Instalacion con modelos por API:

```bash
pip install -r requirements-api.txt
```

## Ejecucion

Modo local:

```bash
python agent.py
```

Modo local usando Ollama remoto en Google Colab por ngrok:

1. Ejecuta `START_COLAB.py` dentro de Google Colab con tu token de ngrok.
2. Copia la URL publica que imprime el script.
3. En tu maquina local configura:

```powershell
$env:KYM_OLLAMA_HOST="https://tu-subdominio.ngrok-free.app"
python agent.py
```

Modo sin IA local:

```bash
python agent.py --no-local
```

## Archivos de dependencias

- `requirements.txt`: dependencias base del servidor.
- `requirements-local.txt`: base + librerias para Ollama y busqueda web.
- `requirements-api.txt`: base + libreria para modelos por API.

## Estructura

- `agent.py`: backend principal.
- `static/`: interfaz web.
- `sessions/`: historial de conversaciones.
- `memory.json`: memoria persistente.
- `START_COLAB.py`: arranque rapido para Google Colab.
