import os
import threading
import time
import subprocess

# 1. Función para instalar Ollama
def install_ollama():
    print("📦 Instalando Ollama...")
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)

# 2. Función para iniciar el servidor Ollama en segundo plano
def start_ollama_serve():
    print("🚀 Iniciando servidor Ollama...")
    os.system("ollama serve > ollama.log 2>&1 &")

# 3. Función para descargar el modelo inicial
def pull_model(model_name="llama3.1"):
    print(f"📥 Descargando modelo {model_name}...")
    subprocess.run(f"ollama pull {model_name}", shell=True)

# 4. Instalación de dependencias de Python
def install_dependencies():
    print("🐍 Instalando dependencias de Python...")
    deps = [
        "fastapi", "uvicorn", "ollama", "duckduckgo-search", 
        "beautifulsoup4", "requests", "python-multipart", 
        "pypdf", "nest_asyncio"
    ]
    subprocess.run(f"pip install {' '.join(deps)}", shell=True)

def main():
    # Detectar si estamos en Google Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if not is_colab:
        print("❌ Este archivo está diseñado para ser ejecutado en GOOGLE COLAB.")
        return

    # Iniciar procesos
    install_ollama()
    start_ollama_serve()
    time.sleep(5)  # Esperar a que el servidor arranque
    pull_model("llama3.1")
    install_dependencies()

    # Configurar el proxy de Google Colab para acceso web
    import nest_asyncio
    from google.colab.output import eval_js
    nest_asyncio.apply()

    # Generar la URL de acceso pública
    proxy_url = eval_js("google.colab.kernel.proxyPort(8000)")
    
    # Configurar variables de entorno para que agent.py sepa que está en Colab
    os.environ["IS_COLAB"] = "true"

    print("\n" + "="*50)
    print("✅ TODO LISTO")
    print(f"🔗 ACCEDE AL AGENTE AQUÍ: {proxy_url}")
    print("⚠️  RECUERDA: La URL del terminal de agent.py contendrá el ?token=... necesario.")
    print("="*50 + "\n")

    # Ejecutar el agente principal
    os.system("python agent.py")

if __name__ == "__main__":
    main()
