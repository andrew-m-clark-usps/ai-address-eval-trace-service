"""Configuration for the eval-trace service."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelEndpointConfig:
    """Configuration for the target AI model endpoint."""

    base_url: str = "http://localhost:8080"
    verify_endpoint: str = "/api/v1/verify"
    health_endpoint: str = "/api/v1/health"
    timeout_seconds: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)
    auth_token: str = ""

    @property
    def verify_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.verify_endpoint}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.health_endpoint}"


@dataclass
class ServiceConfig:
    """Top-level service configuration."""

    model: ModelEndpointConfig = field(default_factory=ModelEndpointConfig)
    output_dir: str = "output"
    data_dir: str = "data/eval_sets"
    run_id: str = ""
    max_concurrent: int = 4
    dashboard_filename: str = "dashboard.html"
    history_filename: str = "run_history.json"

    @classmethod
    def from_env(cls) -> ServiceConfig:
        """Build config from environment variables."""
        model_cfg = ModelEndpointConfig(
            base_url=os.environ.get("MODEL_BASE_URL", "http://localhost:8080"),
            verify_endpoint=os.environ.get("MODEL_VERIFY_ENDPOINT", "/api/v1/verify"),
            health_endpoint=os.environ.get("MODEL_HEALTH_ENDPOINT", "/api/v1/health"),
            timeout_seconds=float(os.environ.get("MODEL_TIMEOUT", "30")),
            auth_token=os.environ.get("MODEL_AUTH_TOKEN", ""),
        )
        if model_cfg.auth_token:
            model_cfg.headers["Authorization"] = f"Bearer {model_cfg.auth_token}"
        return cls(
            model=model_cfg,
            output_dir=os.environ.get("OUTPUT_DIR", "output"),
            data_dir=os.environ.get("DATA_DIR", "data/eval_sets"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ServiceConfig:
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        model_data = data.get("model", {})
        model_cfg = ModelEndpointConfig(**model_data)
        if model_cfg.auth_token:
            model_cfg.headers["Authorization"] = f"Bearer {model_cfg.auth_token}"
        return cls(
            model=model_cfg,
            output_dir=data.get("output_dir", "output"),
            data_dir=data.get("data_dir", "data/eval_sets"),
            max_concurrent=data.get("max_concurrent", 4),
        )
