from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf


# Gann Cycle Repeat Scanner v2
# ------------------------------------------------------------
# What this version adds:
#   - Multi-window scanning
#   - Better historical match scoring
#   - Pivot-aware ranking
#   - Best-shift finder for overlay alignment
#   - Composite from up to 10 selected cycles
#   - Raw / percent / z-score composite modes
#   - Current price stays fixed while composite can be shifted
#   - Yahoo Finance data source
# ============================================================


# -----------------------------
# Config
# -----------------------------
TRADING_DAYS_PER_YEAR = 252
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN",
    "SPY", "QQQ", "IWM", "^GSPC", "^NDX", "GC=F", "CL=F"
]

# Broad practical Gann year-cycle registry.
GANN_YEAR_CYCLES = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 15, 16, 18, 20, 21, 24, 27,
    28, 30, 32, 36, 40, 42, 45, 49, 50, 52,
    54, 56, 60, 63, 64, 72, 80, 84, 90, 96,
    100, 108, 120, 126, 128, 144, 168, 180, 192, 210,
    240, 252, 270, 288, 300, 336, 360,
]

SCAN_WINDOWS_YEARS_DEFAULT = [0.5, 1.0, 1.5, 2.0]


# -----------------------------
# Data containers
# -----------------------------
@dataclass
class CycleMatch:
    cycle_years: float
    bars_offset: int
    window_bars: int
    forecast_bars: int
    score: float
    corr: float
    shape_corr: float
    pivot_score: float
    dir_score: float
    volatility_similarity: float
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    anchor_date: pd.Timestamp
    path_raw: pd.Series
    path_pct: pd.Series
    path_z: pd.Series
    full_path_raw: pd.Series
    full_path_pct: pd.Series
    full_path_z: pd.Series
    shift_score: float = np.nan
    best_shift_bars: int = 0
    meta: Dict[str, float] = field(default_factory=dict)


# -----------------------------
# Utility functions
# -----------------------------
def parse_manual_cycles(text: str) -> List[float]:
    if not text or not text.strip():
        return []
    out: List[float] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
            if val > 0:
                out.append(val)
        except Exception:
            pass
    return sorted(set(out))


_YF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
_YF_API_URLS = [
    "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}",
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
]


def _fetch_yahoo_direct(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch price history via direct Yahoo Finance v8 chart API (no curl_cffi / cookie needed)."""
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())
    params = {"period1": p1, "period2": p2, "interval": "1d", "events": "history", "includeAdjustedClose": "true"}

    session = requests.Session()
    session.headers.update(_YF_HEADERS)

    for url_tpl in _YF_API_URLS:
        url = url_tpl.format(ticker=ticker)
        for attempt in range(3):
            try:
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code != 200:
                    time.sleep(1)
                    continue
                payload = resp.json()
                result = payload.get("chart", {}).get("result")
                if not result:
                    return pd.DataFrame()
                result = result[0]
                timestamps = result.get("timestamp", [])
                indicators = result.get("indicators", {})
                quote = indicators.get("quote", [{}])[0]
                adjclose_list = indicators.get("adjclose", [{}])
                adjclose = adjclose_list[0].get("adjclose", []) if adjclose_list else []

                if not timestamps:
                    return pd.DataFrame()

                idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
                close_vals = adjclose if len(adjclose) == len(timestamps) else quote.get("close", [])
                df = pd.DataFrame(
                    {
                        "Open": quote.get("open", [np.nan] * len(timestamps)),
                        "High": quote.get("high", [np.nan] * len(timestamps)),
                        "Low": quote.get("low", [np.nan] * len(timestamps)),
                        "Close": close_vals if close_vals else [np.nan] * len(timestamps),
                        "Volume": quote.get("volume", [np.nan] * len(timestamps)),
                    },
                    index=idx,
                )
                df = df[~df.index.duplicated(keep="last")].sort_index().dropna(subset=["Close"])
                if not df.empty:
                    return df
            except Exception:
                time.sleep(1)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_price_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Primary: direct Yahoo Finance chart API (no curl_cffi / GDPR cookie needed)
    df = _fetch_yahoo_direct(ticker, start, end)
    if df is not None and not df.empty:
        return df

    # Fallback: yfinance library
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
    except Exception:
        return pd.DataFrame()

    needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[needed].copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df.dropna()


def to_numpy(x: pd.Series) -> np.ndarray:
    return pd.Series(x).astype(float).to_numpy()


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        return 0.0
    if np.allclose(np.nanstd(a), 0) or np.allclose(np.nanstd(b), 0):
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def normalize_start_at_100(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float).copy()
    if len(s) == 0:
        return s
    first = float(s.iloc[0])
    if first == 0:
        return s * np.nan
    return 100.0 * (s / first)


def pct_path(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float)
    first = float(s.iloc[0])
    if first == 0:
        return s * np.nan
    return (s / first - 1.0) * 100.0


def zscore_path(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float)
    sd = float(s.std(ddof=0))
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - float(s.mean())) / sd


def rolling_direction_match(a: pd.Series, b: pd.Series) -> float:
    a1 = pd.Series(a).diff().dropna()
    b1 = pd.Series(b).diff().dropna()
    n = min(len(a1), len(b1))
    if n < 3:
        return 0.0
    sa = np.sign(a1.iloc[:n].to_numpy())
    sb = np.sign(b1.iloc[:n].to_numpy())
    return float((sa == sb).mean())


def pivot_points(series: pd.Series, left: int = 3, right: int = 3) -> pd.DataFrame:
    s = pd.Series(series).astype(float).reset_index(drop=True)
    sv = s.to_numpy(dtype=float)
    # Vectorized rolling comparisons (33× faster than Python loop + iloc)
    left_max = s.rolling(left, min_periods=left).max().shift(1).to_numpy()
    left_min = s.rolling(left, min_periods=left).min().shift(1).to_numpy()
    right_max = s.rolling(right, min_periods=right).max().shift(-right).to_numpy()
    right_min = s.rolling(right, min_periods=right).min().shift(-right).to_numpy()
    highs = (sv > left_max) & (sv >= right_max)
    lows = (sv < left_min) & (sv <= right_min)
    highs = np.where(np.isnan(left_max) | np.isnan(right_max), False, highs)
    lows = np.where(np.isnan(left_min) | np.isnan(right_min), False, lows)
    out = pd.DataFrame({"idx": np.arange(len(s)), "price": s, "high": highs, "low": lows})
    return out


def pivot_similarity(a: pd.Series, b: pd.Series, left: int = 3, right: int = 3, tolerance: int = 2) -> float:
    pa = pivot_points(a, left=left, right=right)
    pb = pivot_points(b, left=left, right=right)
    a_high = pa.index[pa["high"]].tolist()
    b_high = pb.index[pb["high"]].tolist()
    a_low = pa.index[pa["low"]].tolist()
    b_low = pb.index[pb["low"]].tolist()

    def score_side(ref_idx: List[int], cmp_idx: List[int]) -> float:
        if len(ref_idx) == 0:
            return 0.5
        hits = 0
        for i in ref_idx:
            ok = any(abs(i - j) <= tolerance for j in cmp_idx)
            hits += int(ok)
        return hits / len(ref_idx)

    return float((score_side(a_high, b_high) + score_side(a_low, b_low)) / 2.0)


def volatility_similarity(a: pd.Series, b: pd.Series) -> float:
    ra = pd.Series(a).pct_change().dropna()
    rb = pd.Series(b).pct_change().dropna()
    if len(ra) < 5 or len(rb) < 5:
        return 0.0
    sa, sb = float(ra.std(ddof=0)), float(rb.std(ddof=0))
    if sa <= 0 or sb <= 0:
        return 0.0
    ratio = min(sa, sb) / max(sa, sb)
    return float(ratio)


def build_cycle_path(close: pd.Series, end_idx: int, window_bars: int) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.Series]]:
    start_idx = end_idx - window_bars + 1
    if start_idx < 0 or end_idx >= len(close):
        return None
    path = close.iloc[start_idx:end_idx + 1].copy()
    if len(path) != window_bars:
        return None
    return path.index[0], path.index[-1], path


def build_full_segment(close: pd.Series, end_idx: int, window_bars: int, forecast_bars: int) -> Optional[pd.Series]:
    start_idx = end_idx - window_bars + 1
    full_end_idx = min(len(close) - 1, end_idx + forecast_bars)
    if start_idx < 0 or full_end_idx < end_idx:
        return None
    seg = close.iloc[start_idx:full_end_idx + 1].copy()
    if len(seg) < window_bars:
        return None
    return seg


def best_shift_finder(reference_path: pd.Series, candidate_path: pd.Series, max_shift_bars: int = 40) -> Tuple[int, float]:
    ref = pct_path(reference_path).reset_index(drop=True)
    cand = pct_path(candidate_path).reset_index(drop=True)
    n = min(len(ref), len(cand))
    ref = ref.iloc[:n]
    cand = cand.iloc[:n]
    if n < 10:
        return 0, 0.0

    best_shift = 0
    best_score = -1e9

    for shift in range(-max_shift_bars, max_shift_bars + 1):
        if shift >= 0:
            a = ref.iloc[shift:]
            b = cand.iloc[:len(a)]
        else:
            b = cand.iloc[-shift:]
            a = ref.iloc[:len(b)]
        m = min(len(a), len(b))
        if m < 10:
            continue
        a = a.iloc[:m]
        b = b.iloc[:m]
        score = safe_corr(to_numpy(a), to_numpy(b))
        if score > best_score:
            best_score = score
            best_shift = shift

    return int(best_shift), float(best_score)


def scan_repeating_cycles(
    df: pd.DataFrame,
    cycle_years_list: List[float],
    analysis_end_date: pd.Timestamp,
    lookback_years: int,
    windows_years: List[float],
    top_n: int,
    pivot_left: int,
    pivot_right: int,
    forecast_bars: int,
) -> List[CycleMatch]:
    close = df["Close"].copy()
    close = close.loc[:analysis_end_date].dropna()
    if len(close) < 300:
        return []

    analysis_end_idx = len(close) - 1
    results: List[CycleMatch] = []

    for cycle_years in cycle_years_list:
        bars_offset = int(round(cycle_years * TRADING_DAYS_PER_YEAR))
        if bars_offset < 20:
            continue

        anchor_end_idx = analysis_end_idx - bars_offset
        if anchor_end_idx <= 50:
            continue

        for wy in windows_years:
            window_bars = int(round(wy * TRADING_DAYS_PER_YEAR))
            if window_bars < 20:
                continue

            current_info = build_cycle_path(close, analysis_end_idx, window_bars)
            hist_info = build_cycle_path(close, anchor_end_idx, window_bars)
            if current_info is None or hist_info is None:
                continue

            cur_start, cur_end, current_path = current_info
            hist_start, hist_end, hist_path = hist_info
            hist_full_path = build_full_segment(close, anchor_end_idx, window_bars, forecast_bars)
            if hist_full_path is None:
                continue

            earliest_allowed = close.index[-1] - pd.DateOffset(years=lookback_years)
            if hist_end < earliest_allowed:
                continue

            cur_pct = pct_path(current_path).reset_index(drop=True)
            hist_pct = pct_path(hist_path).reset_index(drop=True)
            cur_z = zscore_path(current_path).reset_index(drop=True)
            hist_z = zscore_path(hist_path).reset_index(drop=True)
            full_pct = pct_path(hist_full_path).reset_index(drop=True)
            full_z = zscore_path(hist_full_path).reset_index(drop=True)

            corr = safe_corr(to_numpy(cur_pct), to_numpy(hist_pct))
            shape_corr = safe_corr(to_numpy(cur_z), to_numpy(hist_z))
            piv = pivot_similarity(cur_pct, hist_pct, left=pivot_left, right=pivot_right, tolerance=2)
            dir_score = rolling_direction_match(cur_pct, hist_pct)
            vol_sim = volatility_similarity(current_path, hist_path)
            shift_bars, shift_score = best_shift_finder(current_path, hist_path, max_shift_bars=40)

            score = (
                0.30 * corr
                + 0.25 * shape_corr
                + 0.20 * piv
                + 0.10 * dir_score
                + 0.05 * vol_sim
                + 0.10 * shift_score
            )

            results.append(
                CycleMatch(
                    cycle_years=cycle_years,
                    bars_offset=bars_offset,
                    window_bars=window_bars,
                    forecast_bars=forecast_bars,
                    score=float(score),
                    corr=float(corr),
                    shape_corr=float(shape_corr),
                    pivot_score=float(piv),
                    dir_score=float(dir_score),
                    volatility_similarity=float(vol_sim),
                    start_date=hist_start,
                    end_date=hist_end,
                    anchor_date=hist_end,
                    path_raw=hist_path.copy(),
                    path_pct=hist_pct.copy(),
                    path_z=hist_z.copy(),
                    full_path_raw=hist_full_path.copy(),
                    full_path_pct=full_pct.copy(),
                    full_path_z=full_z.copy(),
                    best_shift_bars=shift_bars,
                    shift_score=shift_score,
                    meta={
                        "current_start": cur_start.value,
                        "current_end": cur_end.value,
                    },
                )
            )

    if not results:
        return []

    # Keep best candidate per (cycle_years, window_bars)
    dedup: Dict[Tuple[float, int], CycleMatch] = {}
    for r in results:
        key = (r.cycle_years, r.window_bars)
        if key not in dedup or r.score > dedup[key].score:
            dedup[key] = r

    out = sorted(dedup.values(), key=lambda x: x.score, reverse=True)
    return out[:top_n]


def build_composite(selected_matches: List[CycleMatch], mode: str = "Percent", use_full_path: bool = True) -> pd.Series:
    if not selected_matches:
        return pd.Series(dtype=float)

    paths = []
    for m in selected_matches:
        if use_full_path:
            raw_path = m.full_path_raw
            pct_series = m.full_path_pct
            z_series = m.full_path_z
        else:
            raw_path = m.path_raw
            pct_series = m.path_pct
            z_series = m.path_z

        if mode == "Raw Indexed":
            p = normalize_start_at_100(raw_path).reset_index(drop=True)
        elif mode == "Z-Score":
            p = z_series.reset_index(drop=True)
        else:
            p = pct_series.reset_index(drop=True)
        paths.append(p)

    max_len = max(len(p) for p in paths)
    arr = np.full((len(paths), max_len), np.nan)
    for i, p in enumerate(paths):
        arr[i, :len(p)] = p.to_numpy(dtype=float)

    comp = np.nanmean(arr, axis=0)
    return pd.Series(comp)


def project_composite_onto_price(
    current_close: pd.Series,
    composite: pd.Series,
    overlay_mode: str,
    vertical_scale: float,
    vertical_offset: float,
    anchor_mode: str = "Last Bar",
    insample_bars: Optional[int] = None,
) -> pd.Series:
    if len(current_close) == 0 or len(composite) == 0:
        return pd.Series(dtype=float)

    comp = pd.Series(composite).astype(float).copy().reset_index(drop=True)
    if insample_bars is None:
        insample_bars = len(current_close)
    insample_bars = min(insample_bars, len(current_close), len(comp))
    comp_in = comp.iloc[:insample_bars].copy()

    if anchor_mode == "First Bar":
        anchor_price = float(current_close.iloc[0])
        anchor_comp_idx = 0
    elif anchor_mode == "Best Fit Window":
        hist_rebase = current_close.iloc[:insample_bars].reset_index(drop=True)
        comp_probe = comp.iloc[:insample_bars].copy()
        if overlay_mode == "Z-Score":
            cstd = float(hist_rebase.std(ddof=0))
            cstd = cstd if cstd > 0 else max(abs(float(hist_rebase.iloc[-1])) * 0.01, 1e-9)
            probe = float(hist_rebase.mean()) + comp_probe * cstd * vertical_scale
        elif overlay_mode == "Percent":
            base0 = float(hist_rebase.iloc[0])
            probe = base0 * (1.0 + (comp_probe / 100.0) * vertical_scale)
        else:
            base0 = float(hist_rebase.iloc[0])
            comp100 = comp_probe / max(abs(float(comp_probe.iloc[0])), 1e-9) * 100.0
            probe = base0 * (comp100 / 100.0)
        diffs = (hist_rebase - pd.Series(probe)).abs()
        anchor_comp_idx = int(diffs.idxmin())
        anchor_price = float(hist_rebase.iloc[anchor_comp_idx])
    else:
        anchor_comp_idx = insample_bars - 1
        anchor_price = float(current_close.iloc[insample_bars - 1])

    if overlay_mode == "Z-Score":
        ref_std = float(current_close.iloc[:insample_bars].std(ddof=0))
        ref_std = ref_std if ref_std > 0 else max(abs(anchor_price) * 0.01, 1e-9)
        raw = anchor_price + (comp - float(comp.iloc[anchor_comp_idx])) * ref_std * vertical_scale + vertical_offset
        return pd.Series(raw)

    if overlay_mode == "Percent":
        anchor_comp_val = float(comp.iloc[anchor_comp_idx])
        rel = (comp - anchor_comp_val) / 100.0
        raw = anchor_price * (1.0 + rel * vertical_scale) + vertical_offset
        return pd.Series(raw)

    # Raw Indexed
    anchor_comp_val = max(abs(float(comp.iloc[anchor_comp_idx])), 1e-9)
    rel = comp / anchor_comp_val
    raw = anchor_price * np.power(np.maximum(rel, 1e-9), vertical_scale) + vertical_offset
    return pd.Series(raw)


def extend_with_future_dates(index: pd.DatetimeIndex, total_length: int) -> pd.DatetimeIndex:
    if len(index) >= total_length:
        return index[-total_length:]
    extra = total_length - len(index)
    future = pd.bdate_range(index[-1] + pd.offsets.BDay(1), periods=extra)
    return pd.DatetimeIndex(list(index) + list(future))


def make_match_label(m: CycleMatch) -> str:
    return (
        f"{m.cycle_years:g}y | win {m.window_bars/TRADING_DAYS_PER_YEAR:.2f}y | "
        f"score {m.score:.3f} | corr {m.corr:.3f} | piv {m.pivot_score:.3f} | "
        f"shift {m.best_shift_bars:+d}"
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Gann Cycle Repeat Scanner v2", layout="wide")
st.title("Gann Cycle Repeat Scanner v2.1")
st.caption("Yahoo prisdata + Gann års-sykluser + multi-window scan + in-sample alignment + ekte forward composite forecast")

with st.sidebar:
    st.header("Data")
    ticker = st.selectbox("Ticker", DEFAULT_TICKERS, index=0)
    custom_ticker = st.text_input("Eller skriv ticker selv", value="")
    ticker = custom_ticker.strip().upper() if custom_ticker.strip() else ticker

    start_date = st.date_input("Historikk start", value=pd.Timestamp("1985-01-01").date())
    end_date = st.date_input("Historikk slutt", value=pd.Timestamp.today().date())

    st.header("Scan")
    analysis_end_date = st.date_input("Analyse cut-off", value=min(pd.Timestamp.today().date(), pd.Timestamp(end_date).date()))
    lookback_years = st.slider("Antall år tilbake å scanne", min_value=5, max_value=80, value=30, step=1)
    top_n_scan = st.slider("Antall scan-resultater", min_value=5, max_value=100, value=40, step=5)
    suggestion_n = st.slider("Beste forslag i UI", min_value=3, max_value=10, value=5, step=1)

    windows_text = st.text_input("Scan-vinduer i år (kommaseparert)", value=", ".join(str(x) for x in SCAN_WINDOWS_YEARS_DEFAULT))
    windows_years = parse_manual_cycles(windows_text)
    if not windows_years:
        windows_years = SCAN_WINDOWS_YEARS_DEFAULT

    pivot_left = st.slider("Pivot left", min_value=1, max_value=10, value=3, step=1)
    pivot_right = st.slider("Pivot right", min_value=1, max_value=10, value=3, step=1)

    st.header("Sykluser")
    use_all_gann = st.checkbox("Bruk hele Gann-registeret", value=True)
    manual_cycles_text = st.text_area(
        "Manuelle syklus-år i tillegg",
        value="",
        height=80,
        help="Eksempel: 3, 7, 10, 20, 30, 60, 90"
    )
    manual_cycles = parse_manual_cycles(manual_cycles_text)

    cycle_registry = sorted(set((GANN_YEAR_CYCLES if use_all_gann else []) + manual_cycles))
    st.caption(f"Aktive sykluser i scan: {len(cycle_registry)}")

    st.header("Composite")
    overlay_mode = st.selectbox("Composite modus", ["Percent", "Raw Indexed", "Z-Score"], index=0)
    forecast_bars = st.slider("Forecast frem i tid (bars)", min_value=5, max_value=252, value=63, step=1)
    max_select = st.slider("Maks antall sykluser i composite", min_value=1, max_value=10, value=5, step=1)
    vertical_scale = st.slider("Vertikal forstørring", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    vertical_offset = st.number_input("Vertikal offset", value=0.0, step=1.0)
    manual_shift = st.slider("Manuell tidsshift for composite (bars)", min_value=-80, max_value=80, value=0, step=1)
    auto_best_shift = st.checkbox("Bruk auto best-shift fra valgte sykluser", value=True)
    anchor_mode = st.selectbox("Anchor modus", ["Last Bar", "First Bar", "Best Fit Window"], index=0)
    show_individual_cycles = st.checkbox("Vis enkeltsykluser", value=True)
    show_normalized_price = st.checkbox("Vis normalisert pris-panel", value=False)
    show_forecast_only = st.checkbox("Vis eget forecast-panel", value=True)
    show_forward_hit_table = st.checkbox("Vis forward score-tabell", value=True)

run_scan = st.button("Kjør v2-scan", type="primary")

if run_scan:
    with st.spinner("Laster Yahoo-data..."):
        df = load_price_history(ticker, str(start_date), str(end_date))

    if df.empty or "Close" not in df.columns:
        st.error("Fant ingen prisdata fra Yahoo for valgt ticker/periode.")
        st.stop()

    analysis_end_ts = pd.Timestamp(analysis_end_date)
    if analysis_end_ts > df.index.max():
        analysis_end_ts = df.index.max()

    if analysis_end_ts < df.index.min():
        st.error("Analyse cut-off er før første datapunkt.")
        st.stop()

    st.success(f"Lastet {ticker}: {len(df)} rader fra {df.index.min().date()} til {df.index.max().date()}")

    n_combos = len(cycle_registry) * len(windows_years)
    with st.spinner(f"Scanner historien for repeterende perioder… ({len(cycle_registry)} sykluser × {len(windows_years)} vinduer = {n_combos} kombinasjoner)"):
        matches = scan_repeating_cycles(
            df=df,
            cycle_years_list=cycle_registry,
            analysis_end_date=analysis_end_ts,
            lookback_years=lookback_years,
            windows_years=windows_years,
            top_n=top_n_scan,
            pivot_left=pivot_left,
            pivot_right=pivot_right,
            forecast_bars=forecast_bars,
        )

    if not matches:
        st.warning("Ingen gyldige match funnet. Prøv kortere vinduer, færre lookback-år eller mer historikk.")
        st.stop()

    st.subheader("Top match scan")
    top_df = pd.DataFrame([
        {
            "cycle_years": m.cycle_years,
            "window_years": round(m.window_bars / TRADING_DAYS_PER_YEAR, 2),
            "score": round(m.score, 4),
            "corr": round(m.corr, 4),
            "shape_corr": round(m.shape_corr, 4),
            "pivot_score": round(m.pivot_score, 4),
            "dir_score": round(m.dir_score, 4),
            "vol_similarity": round(m.volatility_similarity, 4),
            "best_shift_bars": m.best_shift_bars,
            "shift_score": round(m.shift_score, 4),
            "hist_start": m.start_date.date(),
            "hist_end": m.end_date.date(),
            "forecast_bars": m.forecast_bars,
        }
        for m in matches
    ])
    st.dataframe(top_df, use_container_width=True)

    st.subheader("5 beste forslag")
    suggestion_matches = matches[:suggestion_n]
    suggestion_labels = [make_match_label(m) for m in suggestion_matches]
    for i, lbl in enumerate(suggestion_labels, start=1):
        st.write(f"{i}. {lbl}")

    default_selected = suggestion_labels[: min(max_select, len(suggestion_labels))]
    selected_labels = st.multiselect(
        "Velg opptil 10 sykluser til composite",
        options=[make_match_label(m) for m in matches],
        default=default_selected,
        max_selections=max_select,
    )

    selected_matches = [m for m in matches if make_match_label(m) in selected_labels]
    if not selected_matches:
        selected_matches = suggestion_matches[: min(max_select, len(suggestion_matches))]

    composite = build_composite(selected_matches, mode=overlay_mode, use_full_path=True)
    current_window_bars = max(m.window_bars for m in selected_matches)
    current_close = df.loc[:analysis_end_ts, "Close"].tail(current_window_bars).copy()

    if auto_best_shift and selected_matches:
        avg_shift = int(round(np.mean([m.best_shift_bars for m in selected_matches])))
    else:
        avg_shift = 0
    final_shift = avg_shift + manual_shift

    total_overlay_len = min(len(composite), current_window_bars + forecast_bars)
    composite_plot = composite.iloc[:total_overlay_len].copy()
    projected = project_composite_onto_price(
        current_close=current_close,
        composite=composite_plot,
        overlay_mode=overlay_mode,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        anchor_mode=anchor_mode,
        insample_bars=current_window_bars,
    )
    overlay_index = extend_with_future_dates(current_close.index, total_overlay_len)
    projected.index = overlay_index

    shifted_display_y = projected.copy()
    shifted_display_x = overlay_index
    if final_shift != 0:
        shifted_display_y = projected.shift(final_shift)
        shifted_display_x = overlay_index

    st.subheader("Pris + composite overlay")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=current_close.index,
        y=current_close.values,
        mode="lines",
        name=f"{ticker} Close",
        line=dict(width=2),
    ))

    forecast_split_date = current_close.index[-1]

    if show_individual_cycles:
        for m in selected_matches:
            indiv = build_composite([m], mode=overlay_mode, use_full_path=True).iloc[: total_overlay_len]
            indiv_proj = project_composite_onto_price(
                current_close=current_close,
                composite=indiv,
                overlay_mode=overlay_mode,
                vertical_scale=vertical_scale,
                vertical_offset=vertical_offset,
                anchor_mode=anchor_mode,
                insample_bars=current_window_bars,
            )
            indiv_index = overlay_index[:len(indiv_proj)]
            indiv_proj.index = indiv_index
            if final_shift != 0:
                indiv_proj = indiv_proj.shift(final_shift)
            fig.add_trace(go.Scatter(
                x=indiv_index,
                y=indiv_proj.values,
                mode="lines",
                name=f"cycle {m.cycle_years:g}y / win {m.window_bars/TRADING_DAYS_PER_YEAR:.2f}y",
                line=dict(width=1),
                opacity=0.25,
            ))

    insample_mask = shifted_display_x <= forecast_split_date
    forecast_mask = shifted_display_x > forecast_split_date

    fig.add_trace(go.Scatter(
        x=shifted_display_x[insample_mask],
        y=shifted_display_y.loc[insample_mask].values,
        mode="lines",
        name=f"Composite hist ({overlay_mode})",
        line=dict(width=3),
    ))
    fig.add_trace(go.Scatter(
        x=shifted_display_x[forecast_mask],
        y=shifted_display_y.loc[forecast_mask].values,
        mode="lines",
        name=f"Composite forecast ({overlay_mode})",
        line=dict(width=4, dash="dash"),
    ))
    fig.add_vline(x=forecast_split_date, line_dash="dot", line_width=1)
    fig.update_layout(
        height=700,
        xaxis_title="Dato",
        yaxis_title="Pris",
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_normalized_price:
        st.subheader("Normalisert panel")
        norm_price = pct_path(current_close).reset_index(drop=True)
        comp_line = composite.iloc[: len(norm_price)].reset_index(drop=True)
        if overlay_mode == "Raw Indexed":
            comp_line = normalize_start_at_100(comp_line)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=current_close.index,
            y=norm_price.values,
            mode="lines",
            name="Current price %",
            line=dict(width=2),
        ))
        comp_y = comp_line.copy() * vertical_scale + vertical_offset
        comp_y.index = current_close.index[: len(comp_y)]
        if final_shift != 0:
            comp_y = comp_y.shift(final_shift)
        fig2.add_trace(go.Scatter(
            x=current_close.index,
            y=comp_y.values,
            mode="lines",
            name=f"Composite {overlay_mode}",
            line=dict(width=3),
        ))
        fig2.update_layout(height=500, hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

    if show_forecast_only:
        st.subheader("Forecast-only panel")
        forecast_index = overlay_index[current_window_bars:]
        forecast_values = shifted_display_y.iloc[current_window_bars:]
        figf = go.Figure()
        figf.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_values.values,
            mode="lines",
            name="Composite forecast",
            line=dict(width=4, dash="dash"),
        ))
        if len(current_close) > 0:
            figf.add_hline(y=float(current_close.iloc[-1]), line_dash="dot", line_width=1)
        figf.update_layout(height=420, hovermode="x unified", xaxis_title="Dato", yaxis_title="Forecast nivå")
        st.plotly_chart(figf, use_container_width=True)

    if show_forward_hit_table:
        st.subheader("Forward score-tabell")
        forward_rows = []
        for m in selected_matches:
            fullp = m.full_path_pct.reset_index(drop=True)
            ins = m.path_pct.reset_index(drop=True)
            fut = fullp.iloc[len(ins):len(ins) + forecast_bars].reset_index(drop=True)
            if len(fut) < 5:
                forward_slope = np.nan
                forward_return = np.nan
                forward_vol = np.nan
            else:
                forward_slope = float(np.polyfit(np.arange(len(fut)), fut.values, 1)[0])
                forward_return = float(fut.iloc[-1] - fut.iloc[0])
                forward_vol = float(fut.std(ddof=0))
            forward_rows.append({
                "cycle_years": m.cycle_years,
                "window_years": round(m.window_bars / TRADING_DAYS_PER_YEAR, 2),
                "match_score": round(m.score, 4),
                "best_shift": m.best_shift_bars,
                "forward_return_pctpts": None if pd.isna(forward_return) else round(forward_return, 2),
                "forward_slope": None if pd.isna(forward_slope) else round(forward_slope, 4),
                "forward_vol": None if pd.isna(forward_vol) else round(forward_vol, 4),
                "hist_range": f"{m.start_date.date()} -> {m.end_date.date()}",
            })
        st.dataframe(pd.DataFrame(forward_rows), use_container_width=True)

    st.subheader("Valgte sykluser")
    chosen_df = pd.DataFrame([
        {
            "label": make_match_label(m),
            "cycle_years": m.cycle_years,
            "window_years": round(m.window_bars / TRADING_DAYS_PER_YEAR, 2),
            "score": round(m.score, 4),
            "corr": round(m.corr, 4),
            "pivot_score": round(m.pivot_score, 4),
            "dir_score": round(m.dir_score, 4),
            "best_shift_bars": m.best_shift_bars,
            "hist_range": f"{m.start_date.date()} -> {m.end_date.date()}",
        }
        for m in selected_matches
    ])
    st.dataframe(chosen_df, use_container_width=True)

    st.subheader("Diagnostikk")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", ticker)
    c2.metric("Antall scan-match", len(matches))
    c3.metric("Valgte sykluser", len(selected_matches))
    c4.metric("Final shift", final_shift)
    c5.metric("Anchor", anchor_mode)

    st.caption(
        "v2.1 skiller mellom in-sample alignment og forward projection. Anchor-modus kan låses til start, slutt eller best local fit. "
        "Forward-tabellen viser hva hver valgt analog faktisk gjorde etter match-vinduet."
    )
else:
    st.info("Sett parametere i venstremenyen og trykk 'Kjør v2-scan'.")
