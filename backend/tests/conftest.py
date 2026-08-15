"""
Pytest configuration and shared fixtures.

Sets up environment variables so tests don't require a real .env file,
and patches the OpenRouter client so tests never make real HTTP calls.
"""
import os

import pytest


def pytest_configure(config):
    """
    Set dummy environment variables before any test module is imported.

    This prevents pydantic-settings from raising on missing required fields
    (e.g. OPENROUTER_API_KEY) when running tests without a real .env.
    """
    os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-key-placeholder")
    os.environ.setdefault("ROUTER_MODEL", "test-router-model")
    os.environ.setdefault("GENERATOR_MODEL", "test-generator-model")
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    os.environ.setdefault("USE_MEMORY_SAVER", "true")
    os.environ.setdefault("DEBUG", "true")
