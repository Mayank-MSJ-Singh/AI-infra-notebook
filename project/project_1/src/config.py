"""
Configuration Management for ML Model Serving API

This module handles all configuration for the application:
- Environment variables
- Model settings
- Server settings
- Monitoring settings
- Feature flags
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pathlib import Path
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    app_name: str = Field(default="ML Model Serving API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development, staging, production)")

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes (development only)")

    # Model Settings
    model_name: str = Field(default="resnet18", description="Model name to load")
    model_path: Optional[str] = Field(default=None, description="Path to model weights file")
    device: str = Field(default="cpu", description="Device to use (cpu, cuda, cuda:0)")
    batch_size: int = Field(default=1, description="Batch size for inference")
    warmup_iterations: int = Field(default=10, description="Number of warmup iterations on startup")

    # Inference Settings
    max_image_size_mb: int = Field(default=10, description="Maximum image file size in MB")
    allowed_image_types: List[str] = Field(default=["image/jpeg", "image/png", "image/jpg"])
    default_top_k: int = Field(default=5, description="Default number of top predictions")
    confidence_threshold: float = Field(default=0.0, description="Minimum confidence threshold")

    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format (json, text)")

    # Monitoring Settings
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=8000, description="Port for metrics endpoint")

    # Caching Settings
    enable_cache: bool = Field(default=False, description="Enable prediction caching")
    cache_size: int = Field(default=1000, description="Maximum cached predictions")
    cache_ttl_seconds: int = Field(default=300, description="Cache TTL in seconds")

    # Rate Limiting Settings
    enable_rate_limit: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Max requests per window")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")

    # CORS Settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")

    # Health Check Settings
    health_check_interval_seconds: int = Field(default=30, description="Health check interval")

    # --- Validators ---

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v

    @field_validator("port", "metrics_port")
    @classmethod
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        supported = ["resnet18", "resnet50", "mobilenet_v2"]
        if v not in supported:
            raise ValueError(f"Model must be one of: {supported}")
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        import torch
        if v == "cpu":
            return v
        if v.startswith("cuda"):
            if not torch.cuda.is_available():
                raise ValueError("CUDA device specified but CUDA not available")
            if ":" in v:
                parts = v.split(":")
                if parts[0] != "cuda":
                    raise ValueError("Only 'cuda:N' format supported")
                try:
                    int(parts[1])
                except ValueError:
                    raise ValueError("Device ID must be integer")
            return v
        raise ValueError(f"Invalid device '{v}'. Must be 'cpu', 'cuda', or 'cuda:N'")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v

    # --- Helper Methods ---

    def is_production(self) -> bool:
        return self.environment == "production"

    def is_development(self) -> bool:
        return self.environment == "development"

    def get_device(self):
        import torch
        return torch.device(self.device)

    def get_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        return Path("models") / f"{self.model_name}.pth"

    def to_dict(self) -> dict:
        return self.model_dump()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# --- Configuration Loading Functions ---

def load_settings() -> Settings:
    """Load application settings from environment / .env file."""
    try:
        settings = Settings()
        print(f"Loaded configuration:")
        print(f"  Environment: {settings.environment}")
        print(f"  Model: {settings.model_name}")
        print(f"  Device: {settings.device}")
        print(f"  Port: {settings.port}")
        return settings
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        raise


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return load_settings()


def override_settings(**kwargs) -> Settings:
    """Create settings with overrides (useful for testing)."""
    settings_dict = load_settings().model_dump()
    settings_dict.update(kwargs)
    return Settings(**settings_dict)


def get_development_settings() -> Settings:
    """Get development-specific settings."""
    return override_settings(
        environment="development", log_level="DEBUG", reload=True,
        batch_size=1, device="cpu", enable_metrics=False, enable_rate_limit=False,
    )


def get_production_settings() -> Settings:
    """Get production-specific settings."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return override_settings(
        environment="production", log_level="INFO", reload=False,
        batch_size=8, device=device, enable_metrics=True,
        enable_rate_limit=True, enable_cache=True,
    )


def validate_configuration(settings: Settings) -> List[str]:
    """Validate entire configuration. Returns list of errors (empty if valid)."""
    import socket
    import torch

    errors: List[str] = []
    if settings.model_path and not Path(settings.model_path).exists():
        errors.append(f"Model file not found: {settings.model_path}")
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        errors.append("CUDA device specified but CUDA is not available")
    for name, val in [("port", settings.port), ("metrics_port", settings.metrics_port)]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((settings.host, val)) == 0:
                    errors.append(f"{name} {val} is already in use")
        except OSError:
            pass
    for d in ["models", "logs"]:
        if not Path(d).exists():
            errors.append(f"Directory '{d}' does not exist")
    return errors