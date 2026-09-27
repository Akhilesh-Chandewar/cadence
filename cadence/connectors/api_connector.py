"""REST API connector (spec §6): paginated JSON endpoints → canonical schema.

Reads an HTTP GET endpoint, extracts the record list (top level or `records_path`,
dot-separated), follows pagination — either page-number params (`page_param` +
`size_param`) or a next-URL token (`next_path`, evaluated against each page's
JSON) — maps columns, and validates into the canonical schema. `transport` is
injectable for network-free tests (httpx.MockTransport).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from cadence.connectors.base import SchemaValidationError, validate_canonical_df
from cadence.connectors.csv_connector import ColumnMapping

_MAX_PAGES = 1000  # hard stop against infinite next-URL loops


class RESTConnector:
    def __init__(
        self,
        url: str,
        mapping: ColumnMapping | None = None,
        records_path: str | None = None,
        page_param: str | None = None,
        size_param: str | None = None,
        page_size: int = 500,
        next_path: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        transport: Any | None = None,
    ) -> None:
        self.url = url
        self.mapping = mapping or ColumnMapping()
        self.records_path = records_path
        self.page_param = page_param
        self.size_param = size_param
        self.page_size = page_size
        self.next_path = next_path
        self.headers = headers or {}
        self.timeout = timeout
        self.transport = transport  # httpx.MockTransport in tests

    # ------------------------------------------------------------------ fetch
    def _get_json(self, client: Any, url: str, params: dict | None = None) -> dict:
        try:
            resp = client.get(url, params=params, headers=self.headers)
            resp.raise_for_status()
            return resp.json()
        except SchemaValidationError:
            raise
        except Exception as exc:
            raise SchemaValidationError(
                f"API request failed for {url}: {type(exc).__name__}: {exc}"
            ) from exc

    @staticmethod
    def _dig(payload: Any, path: str | None) -> Any:
        if not path:
            return payload
        node = payload
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise SchemaValidationError(f"records_path {path!r} not found in response")
            node = node[part]
        return node

    def _collect(self, client: Any) -> list[dict]:
        records: list[dict] = []
        url, params = self.url, None
        if self.page_param:
            params = {self.page_param: 1}
            if self.size_param:
                params[self.size_param] = self.page_size
        pages = 0
        while url and pages < _MAX_PAGES:
            payload = self._get_json(client, url, params)
            page_records = self._dig(payload, self.records_path)
            if not isinstance(page_records, list):
                raise SchemaValidationError("records must be a JSON list")
            records.extend(page_records)
            pages += 1

            if self.next_path:
                nxt = (
                    self._dig(payload, self.next_path)
                    if self._dig(payload, self.next_path)
                    else None
                )
                url, params = nxt, None
                if url and not str(url).startswith("http"):
                    base = str(self.url).split("?")[0]
                    url = f"{base}{url if url.startswith('/') else '/' + url}"
            elif self.page_param:
                params = {**params, self.page_param: int(params[self.page_param]) + 1}
                url = self.url if page_records else None  # stop on an empty page
            else:
                url = None  # single-shot
        return records

    # ------------------------------------------------------------------- load
    def load(self):
        import httpx

        from cadence.connectors.csv_connector import CSVConnector

        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            records = self._collect(client)
        if not records:
            raise SchemaValidationError(f"API returned no records from {self.url}")

        raw = pd.DataFrame(records)
        canonical = CSVConnector(self.mapping).canonicalize(raw)
        return validate_canonical_df(canonical, source_type="api")
