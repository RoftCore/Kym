import json
import logging
from pathlib import Path

from providers.api_ai import OpenAICompatibleProvider


class ProviderRegistry:
    def __init__(
        self,
        *,
        local_provider,
        default_api_provider=None,
        config_file="providers.json",
        logger=None,
    ):
        self.local_provider = local_provider
        self.default_api_provider = default_api_provider
        self.config_file = Path(config_file)
        self.logger = logger or logging.getLogger(__name__)
        self.custom_api_providers = []
        self.load_config()

    def load_config(self):
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, "r", encoding="utf-8") as file_handle:
                data = json.load(file_handle)
        except Exception as exc:
            self.logger.warning("No se pudo cargar la configuración de proveedores: %s", exc)
            return

        local_config = data.get("local", {})
        self.local_provider.configure(
            enabled=local_config.get("enabled"),
            host=local_config.get("host"),
        )

        self.custom_api_providers = []
        for provider_data in data.get("api_providers", []):
            try:
                provider = OpenAICompatibleProvider(
                    provider_id=provider_data.get("provider_id", ""),
                    model_id=provider_data["model_id"],
                    label=provider_data["label"],
                    base_url=provider_data["base_url"],
                    api_key=provider_data.get("api_key", ""),
                    source_label=provider_data.get("source_label", "API"),
                )
                self.custom_api_providers.append(provider)
            except KeyError as exc:
                self.logger.warning(
                    "Proveedor API ignorado por configuración incompleta (%s).", exc
                )

    def save_config(self):
        data = {
            "local": {
                "enabled": self.local_provider.enabled,
                "host": self.local_provider.host,
            },
            "api_providers": [
                provider.to_config() for provider in self.custom_api_providers
            ],
        }

        with open(self.config_file, "w", encoding="utf-8") as file_handle:
            json.dump(data, file_handle, indent=4, ensure_ascii=False)

    def get_api_providers(self):
        providers = []
        if self.default_api_provider and self.default_api_provider.is_configured():
            providers.append(self.default_api_provider)
        providers.extend(self.custom_api_providers)
        return providers

    def get_all_providers(self):
        return [self.local_provider, *self.get_api_providers()]

    async def list_models(self):
        models = []
        for provider in self.get_all_providers():
            models.extend(await provider.list_models())
        return models

    def get_provider_for_model(self, model_id: str):
        for provider in self.get_api_providers():
            if provider.model_id == model_id:
                return provider
        return self.local_provider

    def mask_key(self, api_key: str):
        if not api_key:
            return ""
        if len(api_key) <= 6:
            return "*" * len(api_key)
        return f"{api_key[:3]}***{api_key[-3:]}"

    def local_summary(self, message: str = ""):
        return {
            "enabled": self.local_provider.enabled,
            "host": self.local_provider.host,
            "remote": self.local_provider.is_remote_host(),
            "ngrok": self.local_provider.is_ngrok_host(),
            "can_autostart": self.local_provider.can_autostart(),
            "message": message,
        }

    def api_summary(self, provider):
        api_key = provider.get_api_key()
        return {
            "provider_id": provider.provider_id,
            "label": provider.label,
            "model_id": provider.model_id,
            "base_url": provider.base_url,
            "source_label": provider.source_label,
            "configured": provider.is_configured(),
            "key_hint": self.mask_key(api_key),
            "built_in": provider is self.default_api_provider,
        }

    def summary(self):
        return {
            "local": self.local_summary(),
            "api_providers": [self.api_summary(provider) for provider in self.get_api_providers()],
        }

    def configure_local(self, *, enabled: bool, host: str = "", autostart: bool = True):
        chosen_host = host.strip() or self.local_provider.host
        self.local_provider.configure(enabled=enabled, host=chosen_host)
        self.save_config()

        message = "IA local desactivada."
        if enabled and self.local_provider.is_ngrok_host():
            message = (
                f"IA local remota por ngrok configurada en {self.local_provider.host}."
            )
        elif enabled and self.local_provider.is_remote_host():
            message = f"IA local remota configurada en {self.local_provider.host}."
        elif enabled and autostart and self.local_provider.can_autostart():
            self.local_provider.ensure_running()
            message = "IA local activada en esta máquina."
        elif enabled:
            message = f"IA local configurada en {self.local_provider.host}."

        return self.local_summary(message=message)

    def add_api_provider(
        self,
        *,
        label: str,
        model_id: str,
        base_url: str,
        api_key: str,
        source_label: str = "API",
    ):
        provider = OpenAICompatibleProvider(
            label=label.strip(),
            model_id=model_id.strip(),
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            source_label=source_label.strip() or "API",
        )
        self.custom_api_providers.append(provider)
        self.save_config()
        return self.api_summary(provider)
