from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from private_gpt.components.web.web_search.providers.specstream import (
    SpecStreamBusy,
    SpecStreamSearchProvider,
)
from private_gpt.settings.settings import Settings


@pytest.fixture
def mock_settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.web_search = MagicMock()
    settings.web_search.enabled = True
    settings.web_search.provider = "specstream"
    settings.specstream = MagicMock()
    settings.specstream.api_key = "api-key"
    settings.specstream.speed = "default"
    settings.specstream.timeout = 30
    return settings


@pytest.fixture
def mock_specstream_json_response() -> dict[str, Any]:
    """Mock JSON response from SpecStream Search API."""
    return {
        "data": {
            "kind": "fact",
            "query": "Example query",
            "answer": None,
            "rows": [],
            "sources": [
                {
                    "url": "https://example.com",
                    "title": "Example Domain",
                    "site": "example.com",
                    "snippet": "Example Domain. This domain is for use in illustrative examples.",
                    "published": "2023-01-15",
                    "cited": True,
                },
                {
                    "url": "https://test.com/page",
                    "title": "Test Page",
                    "site": "test.com",
                    "snippet": "A test page with useful information",
                    "cited": False,
                },
                {"title": "Result without url", "snippet": "skipped"},
            ],
            "took_ms": 1200,
        }
    }


def _mock_session(status: int, body: Any) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=body)

    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_response)
    mock_context.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_context
    return mock_session


class TestSpecStreamProvider:
    @pytest.mark.asyncio
    async def test_make_query_parses_sources(
        self, mock_settings, mock_specstream_json_response
    ):
        provider = SpecStreamSearchProvider(mock_settings)
        mock_session = _mock_session(200, mock_specstream_json_response)

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            results = await provider.make_query("  example query ", num_links=5)

        assert [r.url for r in results] == [
            "https://example.com",
            "https://test.com/page",
        ]
        assert results[0].idx == 1
        assert results[0].title == "Example Domain"
        assert results[0].description.startswith("Example Domain.")
        assert results[0].age == "2023-01-15"
        assert results[1].age == "Unknown"

        _, kwargs = mock_session.get.call_args
        assert mock_session.get.call_args.args[0] == (
            "https://api.specstream.ai/v1/web-search"
        )
        assert kwargs["params"] == {"q": "example query", "num": 5, "speed": "default"}
        assert kwargs["headers"]["Authorization"] == "Bearer api-key"

    @pytest.mark.asyncio
    async def test_fast_speed_and_language_are_sent(
        self, mock_settings, mock_specstream_json_response
    ):
        mock_settings.specstream.speed = "fast"
        provider = SpecStreamSearchProvider(mock_settings)
        mock_session = _mock_session(200, mock_specstream_json_response)

        with patch.object(provider, "_ensure_session", return_value=mock_session):
            await provider.make_query("query", num_links=500, language="es")

        params = mock_session.get.call_args.kwargs["params"]
        assert params["speed"] == "fast"
        assert params["language"] == "es"
        assert params["num"] == 100  # clamped

    @pytest.mark.asyncio
    async def test_invalid_key_raises_value_error(self, mock_settings):
        provider = SpecStreamSearchProvider(mock_settings)
        body = {"error": {"code": "missing_api_key", "message": "No key or token."}}

        with (
            patch.object(
                provider, "_ensure_session", return_value=_mock_session(401, body)
            ),
            pytest.raises(ValueError) as exc,
        ):
            await provider._execute_http_request("query", 10, None, None)

        assert "SpecStream Search API invalid key" in str(exc.value)
        assert "missing_api_key" in str(exc.value)

    @pytest.mark.asyncio
    async def test_empty_balance_raises_value_error(self, mock_settings):
        provider = SpecStreamSearchProvider(mock_settings)
        body = {"error": {"code": "insufficient_balance"}}

        with (
            patch.object(
                provider, "_ensure_session", return_value=_mock_session(402, body)
            ),
            pytest.raises(ValueError) as exc,
        ):
            await provider._execute_http_request("query", 10, None, None)

        assert "insufficient_balance" in str(exc.value)

    @pytest.mark.asyncio
    async def test_busy_raises_retryable_error(self, mock_settings):
        provider = SpecStreamSearchProvider(mock_settings)

        with (
            patch.object(
                provider, "_ensure_session", return_value=_mock_session(503, {})
            ),
            pytest.raises(SpecStreamBusy),
        ):
            await provider._execute_http_request("query", 10, None, None)

    @pytest.mark.asyncio
    async def test_busy_is_retried(self, mock_settings, mock_specstream_json_response):
        provider = SpecStreamSearchProvider(mock_settings)
        provider._execute_http_request = AsyncMock(  # type: ignore[method-assign]
            side_effect=[SpecStreamBusy("busy"), mock_specstream_json_response]
        )

        with patch("asyncio.sleep", new=AsyncMock()):
            results = await provider.make_query("query", num_links=10)

        assert provider._execute_http_request.await_count == 2
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_invalid_key_is_not_retried(self, mock_settings):
        provider = SpecStreamSearchProvider(mock_settings)
        provider._execute_http_request = AsyncMock(  # type: ignore[method-assign]
            side_effect=ValueError("invalid key")
        )

        with pytest.raises(ValueError):
            await provider.make_query("query", num_links=10)

        assert provider._execute_http_request.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_query_raises(self, mock_settings):
        provider = SpecStreamSearchProvider(mock_settings)
        with pytest.raises(ValueError):
            await provider.make_query("   ", num_links=10)

    @pytest.mark.asyncio
    async def test_validate_requires_api_key(self, mock_settings):
        mock_settings.specstream.api_key = ""
        provider = SpecStreamSearchProvider(mock_settings)
        with pytest.raises(ValueError):
            await provider.validate()
