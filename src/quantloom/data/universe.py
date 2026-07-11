"""Resolves the ticker universe to trade: the largest US companies by market cap, sourced from
the SEC's own `company_tickers.json` (its official ticker/CIK index).

Chosen over scraping a "biggest companies" table off a finance site: the SEC file has no row
ceiling (~10,400 tickers as of writing), needs no HTML scraping, so truncating to `stock_number`
is a meaningful "N biggest," not an arbitrary slice.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_SECONDS = 15
_SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# The SEC's fair-access policy requires a descriptive User-Agent identifying the requester --
# unidentified/generic requests risk being rate-limited or blocked. It doesn't validate the
# content, just wants something identifying; override via env var if you hit access issues.
_SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "QuantLoom contact@example.com")


def _normalize_ticker(symbol: str) -> str:
    """Alpaca's API rejects hyphenated share classes (e.g. BRK-B)."""
    return symbol.replace("-", ".")


def resolve_universe(stock_number: int) -> list[str]:
    """The `stock_number` largest US companies by market cap (see module docstring for source
    and ordering), deduplicated."""
    response = requests.get(
        _SEC_COMPANY_TICKERS_URL,
        headers={"User-Agent": _SEC_USER_AGENT},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    tickers = [_normalize_ticker(entry["ticker"]) for entry in response.json().values()]

    deduped = list(dict.fromkeys(tickers))
    if len(deduped) != len(tickers):
        logger.warning(
            "Dropped %d duplicate ticker(s) from the resolved universe", len(tickers) - len(deduped)
        )

    resolved = deduped[:stock_number]
    logger.info("Resolved %d tickers (of %d available)", len(resolved), len(deduped))
    return resolved
