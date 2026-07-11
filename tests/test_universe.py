from __future__ import annotations

from dataclasses import dataclass

import pytest

from quantloom.data import universe as universe_module
from quantloom.data.universe import resolve_universe

_SEC_COMPANY_TICKERS_JSON = {
    "0": {"cik_str": 1, "ticker": "NVDA", "title": "Nvidia"},
    "1": {"cik_str": 2, "ticker": "AAPL", "title": "Apple"},
    "2": {"cik_str": 3, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
    "3": {"cik_str": 2, "ticker": "AAPL", "title": "Apple duplicate row"},
}


@dataclass
class _FakeResponse:
    payload: dict

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self.payload


@dataclass
class _Call:
    url: str
    headers: dict[str, str]


@pytest.fixture
def fake_requests_get(monkeypatch: pytest.MonkeyPatch):
    calls: list[_Call] = []

    def _get(url: str, headers: dict[str, str], timeout: int):
        calls.append(_Call(url, headers))
        return _FakeResponse(_SEC_COMPANY_TICKERS_JSON)

    monkeypatch.setattr(universe_module.requests, "get", _get)
    return calls


def test_resolve_universe_hits_sec_company_tickers(fake_requests_get: list[_Call]) -> None:
    resolve_universe(stock_number=10)
    assert any("sec.gov" in call.url for call in fake_requests_get)


def test_resolve_universe_normalizes_and_dedupes(fake_requests_get: list[_Call]) -> None:
    tickers = resolve_universe(stock_number=10)
    assert tickers == ["NVDA", "AAPL", "BRK.B"]


def test_resolve_universe_preserves_source_order(fake_requests_get: list[_Call]) -> None:
    # the SEC file is empirically ordered by market cap descending -- resolve_universe must not
    # reorder it (e.g. alphabetically), since truncating to stock_number relies on that order
    tickers = resolve_universe(stock_number=2)
    assert tickers == ["NVDA", "AAPL"]


def test_resolve_universe_respects_stock_number(fake_requests_get: list[_Call]) -> None:
    tickers = resolve_universe(stock_number=1)
    assert tickers == ["NVDA"]


def test_resolve_universe_sends_identifying_user_agent(fake_requests_get: list[_Call]) -> None:
    resolve_universe(stock_number=1)
    assert fake_requests_get[0].headers.get("User-Agent")
