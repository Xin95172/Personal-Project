"""Crypto market data download helpers for cointegration research.

This module is adapted from ``Quant/projects/Pairs/get crypto info.ipynb``.
The notebook's executable cells are converted into reusable functions and the
hard-coded local paths/API secrets are intentionally removed.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

try:
    from pybit.unified_trading import HTTP
except ImportError:  # pragma: no cover - optional runtime dependency
    HTTP = None

try:
    from pycoingecko import CoinGeckoAPI
except ImportError:  # pragma: no cover - optional runtime dependency
    CoinGeckoAPI = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional runtime dependency
    tqdm = None


BYBIT_BASE_URL = "https://api.bybit.com"
BINANCE_SPOT_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
DEFAULT_TIMEOUT = 30


def create_bybit_session(
    api_key: str | None = None,
    api_secret: str | None = None,
    testnet: bool = False,
) -> Any:
    """Create a pybit HTTP session.

    Public market endpoints work without credentials. Pass credentials only if
    your local Bybit setup requires them.
    """

    if HTTP is None:
        raise ImportError("pybit is required. Install it with: pip install pybit")

    kwargs: dict[str, Any] = {"testnet": testnet}
    if api_key and api_secret:
        kwargs.update({"api_key": api_key, "api_secret": api_secret})
    return HTTP(**kwargs)


def create_bybit_session_from_config(
    config_path: str | Path,
    testnet: bool = False,
) -> Any:
    """Create a Bybit session from a JSON file with api_key/api_secret fields."""

    with Path(config_path).open("r", encoding="utf-8") as file:
        api_config = json.load(file)

    return create_bybit_session(
        api_key=api_config.get("api_key"),
        api_secret=api_config.get("api_secret"),
        testnet=testnet,
    )


def get_all_bybit_usdt_spot_symbols(timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    """Return all Bybit spot symbols ending with USDT."""

    return _get_bybit_usdt_symbols("spot", timeout=timeout)


def get_all_bybit_perp_symbols(timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    """Return all Bybit USDT perpetual symbols."""

    return _get_bybit_usdt_symbols("linear", timeout=timeout)


def get_price(
    symbols: Iterable[str] | None,
    categories: Iterable[str] | None,
    start_date: datetime,
    end_date: datetime,
    database: str | Path,
    interval: int | str = 1,
    session: Any | None = None,
    timezone: str = "Asia/Taipei",
    sleep_seconds: float = 0.5,
) -> None:
    """Download Bybit kline data and save one Parquet per symbol/category.

    Empty ``symbols`` means all Bybit spot USDT pairs. Empty ``categories``
    means both ``spot`` and ``linear``.
    """

    bybit_session = session or create_bybit_session()
    target_symbols = list(symbols or get_all_bybit_usdt_spot_symbols())
    target_categories = list(categories or ["spot", "linear"])
    database = Path(database)

    for symbol in target_symbols:
        for category in target_categories:
            try:
                klines_data = _fetch_bybit_klines(
                    bybit_session=bybit_session,
                    symbol=symbol,
                    category=category,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
                    sleep_seconds=sleep_seconds,
                )
                if not klines_data:
                    print(f"{symbol} {category}: no kline data")
                    continue

                header = [
                    "Timestamp",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Turnover",
                ]
                df = pd.DataFrame(reversed(klines_data), columns=header)
                df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
                df["Timestamp"] = df["Timestamp"].dt.tz_convert(timezone)
                df.set_index("Timestamp", inplace=True)

                save_dir = database / category
                save_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(save_dir / f"{symbol}.parquet")
            except Exception as exc:
                print(symbol, category, exc)


def get_funding_rate(
    symbols: Iterable[str] | None,
    start_date: datetime,
    end_date: datetime,
    database: str | Path,
    session: Any | None = None,
    timezone: str = "Asia/Taipei",
    sleep_seconds: float = 0.5,
) -> None:
    """Download Bybit linear funding rate history and save Parquet files."""

    bybit_session = session or create_bybit_session()
    target_symbols = list(symbols or get_all_bybit_perp_symbols())
    database = Path(database)

    for symbol in target_symbols:
        try:
            funding_rate_data = _fetch_bybit_funding_rates(
                bybit_session=bybit_session,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                sleep_seconds=sleep_seconds,
            )
            if not funding_rate_data:
                print(f"{symbol}: no funding rate data")
                continue

            df = pd.DataFrame(reversed(funding_rate_data))
            df["fundingRateTimestamp"] = pd.to_numeric(
                df["fundingRateTimestamp"],
                errors="coerce",
            )
            df["fundingRateTimestamp"] = pd.to_datetime(
                df["fundingRateTimestamp"],
                unit="ms",
                utc=True,
            )
            df["fundingRateTimestamp"] = df["fundingRateTimestamp"].dt.tz_convert(
                timezone,
            )
            df.set_index("fundingRateTimestamp", inplace=True)

            save_dir = database / "funding_rate"
            save_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(save_dir / f"{symbol}.parquet")
        except Exception as exc:
            print(symbol, exc)


def fetch_sector_map(
    base_path: str | Path,
    categories_to_process: Iterable[str] = ("spot", "linear", "funding_rate"),
    sleep_seconds: float = 1.2,
) -> None:
    """Build CoinGecko sector map JSON files for the top 10 market sectors."""

    if CoinGeckoAPI is None:
        raise ImportError("pycoingecko is required. Install it with: pip install pycoingecko")

    base_path = Path(base_path)
    meta_dir = base_path / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    cg = CoinGeckoAPI()

    print("Fetching CoinGecko category list...")
    try:
        # get_coins_categories() returns list of categories with market data
        categories = cg.get_coins_categories()
    except Exception as exc:
        print(f"Cannot fetch CoinGecko category data: {exc}")
        return

    # Sort categories by market cap (descending) and get the top 10
    # Handle cases where market_cap might be None
    top_10_categories = sorted(
        categories, 
        key=lambda x: x.get("market_cap") or 0, 
        reverse=True
    )[:10]

    print(f"Selected Top 10 Categories by Market Cap:")
    for i, cat in enumerate(top_10_categories, 1):
        print(f"  {i}. {cat['name']}")

    sector_map: dict[str, dict[str, Any]] = {}
    iterator = tqdm(top_10_categories, desc="Fetching Top 10 sectors") if tqdm else top_10_categories

    for category in iterator:
        cat_id = category["id"]
        cat_name = category["name"]
        try:
            # Fetch all coins in this category (up to 250)
            coins = cg.get_coins_markets(vs_currency="usd", category=cat_id, per_page=250)
            
            # Format to uppercase symbol + USDT to match exchange standards
            symbols = [coin["symbol"].upper() + "USDT" for coin in coins]
            
            if symbols:
                sector_map[cat_name] = {"id": cat_id, "symbols": sorted(set(symbols))}
                
            time.sleep(sleep_seconds)
        except Exception as exc:
            print(f"Cannot fetch category {cat_name}: {exc}")

    # Generate the global metadata file
    sector_file = meta_dir / "top10_market_sector_map.json"
    with sector_file.open("w", encoding="utf-8") as file:
        json.dump(sector_map, file, indent=2, ensure_ascii=False)
        
    print(f"Successfully saved Top 10 sector map to {sector_file.name}")



def get_binance_futures_klines(
    symbols: Iterable[str],
    start_date: datetime,
    end_date: datetime,
    database: str | Path,
    interval: str = "4h",
    sleep_seconds: float = 15,
) -> None:
    """Download Binance USDT-M futures klines via public REST API."""

    save_dir = Path(database) / "futures"
    save_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        klines_raw = _fetch_binance_klines(
            base_url=BINANCE_FUTURES_BASE_URL,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            limit=1000,
            sleep_seconds=sleep_seconds,
        )
        df = _format_binance_klines(klines_raw)
        df.to_parquet(save_dir / f"{symbol}.parquet", index=False)


def get_binance_spot_klines(
    symbols: Iterable[str],
    start_date: datetime,
    end_date: datetime,
    database: str | Path,
    interval: str = "4h",
    sleep_seconds: float = 30,
) -> None:
    """Download Binance spot klines via public REST API."""

    save_dir = Path(database) / "spot"
    save_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        klines_raw = _fetch_binance_klines(
            base_url=BINANCE_SPOT_BASE_URL,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            limit=500,
            sleep_seconds=sleep_seconds,
        )
        df = _format_binance_klines(klines_raw)
        df.to_parquet(save_dir / f"{symbol}.parquet", index=False)


def get_binance_funding_rate(
    symbols: Iterable[str],
    start_date: datetime,
    end_date: datetime,
    database: str | Path,
    sleep_seconds: float = 1,
) -> None:
    """Download Binance funding rates via public REST API."""

    save_dir = Path(database) / "funding_rate"
    save_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        funding_rates = []
        start_ms = _datetime_to_milliseconds(start_date)
        end_ms = _datetime_to_milliseconds(end_date)

        while start_ms < end_ms:
            response = requests.get(
                f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate",
                params={
                    "symbol": symbol,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 1000,
                },
                timeout=DEFAULT_TIMEOUT,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break

            funding_rates.extend(batch)
            start_ms = int(batch[-1]["fundingTime"]) + 1
            time.sleep(sleep_seconds)

        df = pd.DataFrame(funding_rates)
        if not df.empty:
            df["open_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df = df[["open_time", "fundingRate"]]
        df.to_parquet(save_dir / f"{symbol}.parquet", index=False)


def get_all_binance_spot_symbols(timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    """Return all trading Binance spot symbols ending with USDT."""
    response = requests.get(f"{BINANCE_SPOT_BASE_URL}/api/v3/exchangeInfo", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
    ]


def get_all_binance_futures_symbols(timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    """Return all trading Binance USDT-M perpetual symbols."""
    response = requests.get(f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/exchangeInfo", timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return [
        s["symbol"] for s in data["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING" and s["contractType"] == "PERPETUAL"
    ]


def _get_bybit_usdt_symbols(category: str, timeout: int = DEFAULT_TIMEOUT) -> list[str]:
    symbols: list[str] = []
    cursor = None

    while True:
        params = {"category": category, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            f"{BYBIT_BASE_URL}/v5/market/instruments-info",
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload["result"]
        symbols.extend(item["symbol"] for item in result["list"])

        cursor = result.get("nextPageCursor")
        if not cursor:
            break

    return [symbol for symbol in symbols if symbol.endswith("USDT")]


def _fetch_bybit_klines(
    bybit_session: Any,
    symbol: str,
    category: str,
    start_date: datetime,
    end_date: datetime,
    interval: int | str,
    sleep_seconds: float,
) -> list[list[str]]:
    start_date_ts = _datetime_to_milliseconds(start_date)
    end_date_ts = _datetime_to_milliseconds(end_date)
    klines_data: list[list[str]] = []

    while True:
        response = bybit_session.get_kline(
            category=category,
            symbol=symbol,
            interval=str(interval),
            start=start_date_ts,
            end=end_date_ts,
            limit=1000,
        )
        klines = response["result"]["list"]
        if not klines:
            break

        klines_data.extend(klines)
        end_date_ts = int(klines[-1][0]) - 1

        if len(klines) < 1000 or end_date_ts < start_date_ts:
            break
        time.sleep(sleep_seconds)

    return klines_data


def _fetch_bybit_funding_rates(
    bybit_session: Any,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    start_date_ts = _datetime_to_milliseconds(start_date)
    end_date_ts = _datetime_to_milliseconds(end_date)
    funding_rate_data: list[dict[str, Any]] = []

    while True:
        response = bybit_session.get_funding_rate_history(
            category="linear",
            symbol=symbol,
            startTime=start_date_ts,
            endTime=end_date_ts,
            limit=200,
        )
        funding_rate = response["result"]["list"]
        if not funding_rate:
            break

        funding_rate_data.extend(funding_rate)
        end_date_ts = int(funding_rate[-1]["fundingRateTimestamp"]) - 1

        if len(funding_rate) < 200 or end_date_ts < start_date_ts:
            break
        time.sleep(sleep_seconds)

    return funding_rate_data


def _fetch_binance_klines(
    base_url: str,
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
    limit: int,
    sleep_seconds: float,
) -> list[list[Any]]:
    klines_raw: list[list[Any]] = []
    start_ms = _datetime_to_milliseconds(start_date)
    end_ms = _datetime_to_milliseconds(end_date)

    while start_ms < end_ms:
        response = requests.get(
            f"{base_url}/api/v3/klines" if base_url == BINANCE_SPOT_BASE_URL else f"{base_url}/fapi/v1/klines",
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break

        klines_raw.extend(batch)
        start_ms = int(batch[-1][0]) + 1
        time.sleep(sleep_seconds)

    return klines_raw


def _format_binance_klines(klines_raw: list[list[Any]]) -> pd.DataFrame:
    headers = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_vol",
        "#trade",
        "taker_buy_vol",
        "taker_buy_quote_vol",
        "ignore",
    ]
    df = pd.DataFrame(klines_raw, columns=headers)
    if df.empty:
        return df

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "#trade",
        "taker_buy_vol",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.drop(columns=["ignore", "quote_vol", "taker_buy_quote_vol"], inplace=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    df["week"] = df["open_time"].dt.dayofweek
    return df


def _datetime_to_milliseconds(value: datetime) -> int:
    return int(value.timestamp() * 1000)


if __name__ == "__main__":
    # Example:
    # base = Path("data/crypto_database")
    # start = datetime(2022, 1, 1)
    # end = datetime.now()
    # get_price([], [], start, end, base, interval=60)
    # get_funding_rate([], start, end, base)
    pass
