import asyncio
import logging
from typing import Any

import aiohttp
from aiohttp import ClientResponse
from injector import inject, singleton

from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider
from private_gpt.settings.settings import Settings
from private_gpt.utils.retry import retry

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_JITTER = (1.0, 5.0)
_BASE_URL = "https://api.specstream.ai"
_WEB_SEARCH_ENDPOINT = "/v1/web-search"
_DELAY = 1.1
_BACKOFF = 2
_MAX_NUM = 100  # SpecStream allows num 1-100
_MAX_QUERY_LENGTH = 2000


class SpecStreamBusy(Exception):
    """Raised on HTTP 503: the per-minute limit was used or search is busy."""

    pass


@singleton
class SpecStreamSearchProvider(BaseWebSearchProvider):
    _session: aiohttp.ClientSession | None = None

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = self._settings.specstream.api_key
        self._speed = self._settings.specstream.speed

    async def make_query(
        self,
        query: str,
        num_links: int = 10,
        language: str | None = None,
        country: str | None = None,
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        # 1. Check params
        query, num_links = self._validate_query_params(query, num_links)

        # 2. Make the request
        response_data = await self._execute_with_retry(
            query, num_links, language, country
        )

        # 3. Parse response
        return await asyncio.to_thread(self._parse_response, response_data)

    async def validate(self) -> None:
        if not self._api_key or not self._api_key.strip():
            raise ValueError("SpecStream API key is not configured")

    def _validate_query_params(self, query: str, num_links: int) -> tuple[str, int]:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        if len(query.strip()) > _MAX_QUERY_LENGTH:
            raise ValueError(
                f"Search query cannot be longer than {_MAX_QUERY_LENGTH} characters"
            )

        normalized_num_links = max(1, min(_MAX_NUM, num_links))
        if normalized_num_links != num_links:
            logger.warning(
                f"Num_links {num_links} outside valid range [1,{_MAX_NUM}], "
                f"clamped to {normalized_num_links}"
            )

        return query, normalized_num_links

    async def _execute_http_request(
        self,
        query: str,
        num_links: int,
        language: str | None,
        country: str | None,
    ) -> Any:
        """Execute HTTP request to SpecStream Search API.

        This method performs the actual HTTP call without retry logic,
        allowing for easier testing and mocking.
        """
        session = await self._ensure_session()
        async with session.get(
            f"{_BASE_URL}{_WEB_SEARCH_ENDPOINT}",
            params=self._build_request_params(query, num_links, language, country),
            headers=self._build_request_headers(),
            timeout=aiohttp.ClientTimeout(total=self._settings.specstream.timeout),
        ) as response:
            try:
                response_data = await response.json(content_type=None)
            except (aiohttp.ContentTypeError, ValueError):
                response_data = {}

            await asyncio.to_thread(self._check_response, response, response_data)
            return response_data

    @retry(
        exceptions=(SpecStreamBusy, aiohttp.ClientConnectionError, TimeoutError),
        is_async=True,
        delay=_DELAY,
        backoff=_BACKOFF,
        tries=_MAX_RETRIES,
        jitter=_JITTER,
        logger=logger,
    )
    async def _execute_with_retry(
        self,
        query: str,
        num_links: int,
        language: str | None,
        country: str | None,
    ) -> Any:
        """Execute SpecStream Search with retry logic for transient errors.

        Only busy responses (503) and connection errors are retried; a bad
        key, an empty balance or an invalid query fail at once.
        """
        logger.debug(f"Executing SpecStream Search: query='{query}'")
        response_data = await self._execute_http_request(
            query, num_links, language, country
        )
        logger.debug(f"SpecStream Search successful: query='{query}'")
        return response_data

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            enabled_proxy = self._settings.server.network.proxy.enabled
            self._session = aiohttp.ClientSession(trust_env=enabled_proxy)
        return self._session

    def _extract_error_message(self, response_data: Any) -> str:
        error = response_data.get("error") if isinstance(response_data, dict) else None
        if isinstance(error, dict):
            parts = []
            code = error.get("code")
            message = error.get("message")
            if code:
                parts.append(f"code={code}")
            if message:
                parts.append(f"message={message}")
            if parts:
                return "; ".join(parts)
        if error:
            return str(error)
        return "Unknown error"

    def _check_response(self, response: ClientResponse, response_data: Any) -> None:
        status_code = response.status
        if status_code < 400:
            if not isinstance(response_data, dict) or not isinstance(
                response_data.get("data"), dict
            ):
                raise ValueError("SpecStream Search API returned an unexpected body")
            return

        error_message = self._extract_error_message(response_data)
        if status_code == 503:
            logger.debug(f"SpecStream Search API busy: {error_message}")
            raise SpecStreamBusy(f"SpecStream Search API busy: {error_message}")
        elif status_code == 401:
            raise ValueError(f"SpecStream Search API invalid key ({error_message})")
        elif status_code in (402, 403):
            raise ValueError(
                f"SpecStream Search API balance or spending limit reached ({error_message})"
            )
        elif status_code == 422:
            raise ValueError(f"SpecStream Search API invalid query ({error_message})")
        else:
            raise Exception(
                f"SpecStream Search API error ({status_code}): {error_message}"
            )

    def _build_request_params(
        self,
        query: str,
        num_links: int,
        language: str | None,
        country: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "q": query.strip(),
            "num": num_links,
            "speed": self._speed,
        }
        if language:
            params["language"] = language
        if country:
            params["country"] = country

        return params

    def _build_request_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def _parse_response(self, data: dict[str, Any]) -> list[WebSearchResult]:
        results = []
        sources = (data.get("data") or {}).get("sources") or []

        for item in sources:
            try:
                url = item.get("url")
                if url:
                    results.append(
                        WebSearchResult(
                            idx=len(results) + 1,
                            title=item.get("title") or "No title",
                            url=url,
                            description=item.get("snippet") or "",
                            age=item.get("published") or "Unknown",
                        )
                    )
            except Exception as e:
                logger.warning(f"Error parsing result: {e}")
                continue

        return results

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Closed aiohttp ClientSession")
