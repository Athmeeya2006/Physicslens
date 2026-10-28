"""Model allowlist from config/models.yaml. Anything not listed is refused."""

from __future__ import annotations

import os
from typing import Any

from .errors import CostGuardError, ModelNotAllowed


class Allowlist:
    def __init__(self, models_cfg: dict[str, Any]):
        self._provider_of: dict[str, str] = {}
        self._provider_cfg: dict[str, dict[str, Any]] = {}
        for provider, cfg in models_cfg.items():
            self._provider_cfg[provider] = cfg
            for model_id in cfg.get("models") or {}:
                self._provider_of[model_id] = provider

    def provider_for(self, model: str) -> str:
        try:
            return self._provider_of[model]
        except KeyError:
            raise ModelNotAllowed(
                f"Model {model!r} is not in config/models.yaml; refusing the call."
            ) from None

    def model_cfg(self, model: str) -> dict[str, Any]:
        return (self._provider_cfg[self.provider_for(model)].get("models") or {}).get(model) or {}

    def models(self, provider: str) -> list[str]:
        return [m for m, p in self._provider_of.items() if p == provider]

    def base_url(self, provider: str) -> str:
        cfg = self._provider_cfg[provider]
        if "base_url" in cfg:
            return cfg["base_url"]
        # Self-hosted servers have no fixed address; the machine running them sets it.
        url = os.environ.get(cfg["base_url_env"], "").strip()
        if not url:
            raise CostGuardError(f"{cfg['base_url_env']} is not set: no {provider} server to call.")
        return url

    def api_key_env(self, provider: str) -> str:
        return self._provider_cfg[provider]["api_key_env"]
