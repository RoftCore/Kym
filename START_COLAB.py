import argparse
import os
import socket
import subprocess
import time


DEFAULT_MODEL = "llama3.1"
OLLAMA_PORT = 11434


parser = argparse.ArgumentParser(description="Launch Ollama on Google Colab and expose it via ngrok.")
parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo inicial para descargar en Ollama")
parser.add_argument(
    "--ngrok-token",
    default=os.getenv("NGROK_AUTHTOKEN") or os.getenv("NGROK_TOKEN") or "",
    help="Token de autenticación de ngrok",
)
args, _ = parser.parse_known_args()


def install_ollama():
    print("Installing Ollama...")
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)


def wait_for_port(host: str, port: int, timeout_seconds: int = 30):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                return True
        time.sleep(1)
    return False


def start_ollama_serve():
    if wait_for_port("127.0.0.1", OLLAMA_PORT, timeout_seconds=2):
        print("Ollama server already running.")
        return

    print("Starting Ollama server...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not wait_for_port("127.0.0.1", OLLAMA_PORT, timeout_seconds=30):
        raise RuntimeError("Ollama did not start correctly on port 11434.")


def pull_model(model_name):
    print(f"Downloading model {model_name}...")
    subprocess.run(["ollama", "pull", model_name], check=True)


def install_ngrok_dependencies():
    print("Installing ngrok dependencies...")
    subprocess.run("pip install pyngrok", shell=True, check=True)


def open_ngrok_tunnel(authtoken: str):
    if not authtoken:
        raise RuntimeError(
            "Missing ngrok token. Set NGROK_AUTHTOKEN or pass --ngrok-token."
        )

    from pyngrok import ngrok

    ngrok.set_auth_token(authtoken)
    tunnel = ngrok.connect(OLLAMA_PORT, bind_tls=True)
    return tunnel


def ensure_colab():
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("This file is intended to be run in Google Colab.") from exc


def print_next_steps(public_url: str, model_name: str):
    print("\n" + "=" * 60)
    print("OLLAMA ON COLAB IS READY")
    print(f"ngrok URL: {public_url}")
    print(f"Model loaded: {model_name}")
    print("")
    print("Start your local app with one of these options:")
    print(f"  PowerShell (current session): $env:KYM_OLLAMA_HOST=\"{public_url}\"")
    print(f"  PowerShell (persistent): setx KYM_OLLAMA_HOST \"{public_url}\"")
    print("  Then run locally: python agent.py")
    print("=" * 60 + "\n")


def main():
    ensure_colab()
    install_ollama()
    start_ollama_serve()
    pull_model(args.model)
    install_ngrok_dependencies()

    tunnel = open_ngrok_tunnel(args.ngrok_token)
    print_next_steps(tunnel.public_url, args.model)

    print("Keeping Colab runtime alive. Stop the cell to close ngrok and Ollama.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping ngrok tunnel...")
        try:
            from pyngrok import ngrok

            ngrok.kill()
        except Exception:
            pass


if __name__ == "__main__":
    main()
