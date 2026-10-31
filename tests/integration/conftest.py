"""Integration tests run against the real services from ``docker compose up -d``.

Connection settings come from the environment / .env, exactly as the app reads them.
"""

import pytest

from rag_platform.core.config import Settings


@pytest.fixture(scope="session")
def live_settings() -> Settings:
    return Settings()
