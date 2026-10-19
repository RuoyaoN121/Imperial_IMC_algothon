"""
IMCity Algothon — Trading Bot v3
=================================
Key improvement: MARKET-DERIVED EDGE CALCULATION

Instead of naively comparing theo vs best bid/ask, we now:
1. Track all recent trades per product (from SSE stream + polling)
2. Compute market consensus via:
   - VWAP (volume-weighted average price)
   - Exponentially-weighted mean (recent trades count more)
   - Gaussian fit (mean + std dev of trade distribution)
3. Edge = |theo - market_consensus|
4. Market std dev informs our quoting width (wider if market is noisy)
5. Edge magnitude drives position sizing

The edge calculation pipeline:
┌─────────────────────────────────────────────────────────┐
│  Raw trade stream → TradeTracker (per product)          │
│       ↓                                                 │
│  Compute: VWAP, EW mean, Gaussian(μ, σ)                │
│       ↓                                                 │
│  market_consensus = weighted blend of the above         │
│  market_uncertainty = σ from Gaussian (or EW std)       │
│       ↓                                                 │
│  edge = theo - market_consensus                         │
│  edge_zscore = edge / market_uncertainty                │
│       ↓                                                 │
│  Sizing: bigger when |edge_zscore| is large             │
│  Quoting: wider when market_uncertainty is large        │
└─────────────────────────────────────────────────────────┘
"""

import time
import math
import json
import requests
import pandas as pd
import numpy as np
from threading import Thread, Lock
from collections import deque
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.stats import norm
from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side, Trade

# ============================================================
# CONFIG
# ============================================================
class Config:
    EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
    USERNAME = "particle accelerators"       # TODO: fill in
    PASSWORD = "lebronisgoat"       # TODO: fill in
    RAPID_API_KEY = "6c94f344c9msh1c1f381bea99cbdp1e310fjsn9bd2fcb93d70" 

    START_TIME = "2026-02-28 12:00:00"
    SETTLEMENT_TIME = "2026-03-01 12:00:00"

    # Market making
    BASE_QUOTE_WIDTH_PCT = 0.004
    MIN_QUOTE_WIDTH = 2
    MAX_POSITION = 95              # use more of the ±100 limit
    SKEW_FACTOR = 3.0              # aggressive inventory skew

    # Sizing — CRITICAL: was 1/1/1, completely killed PnL
    MIN_ORDER_SIZE = 1
    MAX_ORDER_SIZE = 25            # ±100 limit means we need real size
    BASE_ORDER_SIZE = 12            # passive quotes should be meaningful
    LOTTERY_ORDER_SIZE = 5         # [NEW] Small size for deep book quotes
    LOTTERY_WIDTH_MULT = 5.0

    # Arbitrage
    ARB_MIN_PROFIT = 2             # capture smaller arb edges (was 3)
    ARB_MAX_SIZE = 10              # arb is risk-free, go bigger (was 5)

    # Rate limiting
    MIN_SECONDS_BETWEEN_REQUOTES = 1.2
    THEO_CHANGE_THRESHOLD_PCT = 0.0005

    # Data refresh
    WEATHER_REFRESH_SECS = 90
    TIDE_REFRESH_SECS = 90
    FLIGHT_REFRESH_SECS = 600

    # Trade tracking
    TRADE_HISTORY_WINDOW = 600     # 10 min of trades for better stats
    MIN_TRADES_FOR_STATS = 5
    MARKET_TRADE_POLL_SECS = 12


    # Edge calculation weights
    VWAP_WEIGHT = 0.4
    EW_MEAN_WEIGHT = 0.4
    GAUSSIAN_MEAN_WEIGHT = 0.2

    # Scoring: Group A (×4 multiplier) — prioritize these for max points
    GROUP_A = {"WX_SPOT", "LHR_COUNT", "LON_ETF"}
    GROUP_A_SIZE_MULT = 1.5        # trade 50% more aggressively on 4× products

    # Confidence per product — calibrated against observed market gaps
    # High confidence = trust our theo. Low confidence = blend with market.
    PRODUCT_CONFIDENCE = {
        "WX_SPOT": 0.5,       # perfectly aligned with market, forecast reliable
        "WX_SUM": 0.7,       # accumulation of reliable weather data
        "TIDE_SPOT": 0.4,     # tidal predictions are scientific, 4% gap OK
        "TIDE_SWING": 0.6,    # strangle payoff harder to predict
        "LHR_COUNT": 0.4,    # 15% off market — trust market more
        "LHR_INDEX": 0.2,    # 162% off market — our model is broken here
        "LON_ETF": 0.65,       # cascades from components
        "LON_FLY": 0.4,       # cascades from ETF
    }

    # Theo-market blending: effective_theo = conf * our_theo + (1-conf) * market
    # For low-confidence products, this prevents trading against the market
    BLEND_THEO_WITH_MARKET = True


# ============================================================
# 1. TRADE TRACKER — collects and analyzes recent trades
# ============================================================
@dataclass
class TradeRecord:
    """Lightweight trade record for analysis."""
    timestamp: float    # unix timestamp (monotonic-ish)
    price: float
    volume: int
    product: str


class TradeTracker:
    """Collects recent trades per product and computes market statistics.

    For each product, maintains a sliding window of recent trades and
    computes three views of market consensus:

    1. VWAP — volume-weighted average price
       Big trades pull the average more. If someone trades 10@5000,
       that's a stronger signal than 1@4800.

    2. Exponentially-weighted mean — recent trades count more
       Uses a half-life so the most recent trades dominate.
       If market was at 5000 ten minutes ago but last 5 trades
       were at 5100, the EW mean is closer to 5100.

    3. Gaussian fit — mean and standard deviation
       Fits N(μ, σ) to the trade price distribution.
       μ = where the market thinks fair value is
       σ = how much disagreement/noise there is
       Low σ + far from theo = strong edge signal
       High σ = market is uncertain, be careful

    The final market_consensus is a weighted blend of all three.
    """

    def __init__(self, window_seconds=300):
        self.window = window_seconds
        self._trades: dict[str, deque[TradeRecord]] = {}
        self._lock = Lock()

    def add_trade(self, product: str, price: float, volume: int):
        """Add a trade to the tracker."""
        with self._lock:
            if product not in self._trades:
                self._trades[product] = deque()
            self._trades[product].append(TradeRecord(
                timestamp=time.time(),
                price=price,
                volume=volume,
                product=product,
            ))
            self._prune(product)

    def add_market_trades(self, trades_list):
        """Bulk add from market trade poll."""
        with self._lock:
            for t in trades_list:
                product = t.product if hasattr(t, 'product') else t.get('product', '')
                price = float(t.price if hasattr(t, 'price') else t.get('price', 0))
                volume = int(t.volume if hasattr(t, 'volume') else t.get('volume', 0))
                if product and price > 0:
                    if product not in self._trades:
                        self._trades[product] = deque()
                    self._trades[product].append(TradeRecord(
                        timestamp=time.time(),
                        price=price,
                        volume=volume,
                        product=product,
                    ))
            for product in self._trades:
                self._prune(product)

    def _prune(self, product):
        """Remove trades older than the window."""
        cutoff = time.time() - self.window
        q = self._trades.get(product)
        if q:
            while q and q[0].timestamp < cutoff:
                q.popleft()

    def get_stats(self, product) -> dict | None:
        """Compute market statistics for a product.

        Returns dict with:
            vwap: volume-weighted average price
            ew_mean: exponentially-weighted mean (half-life = window/4)
            ew_std: exponentially-weighted std dev
            gauss_mean: Gaussian fit mean
            gauss_std: Gaussian fit std dev
            consensus: weighted blend of the above
            n_trades: number of trades in window
            total_volume: total volume traded
        """
        with self._lock:
            self._prune(product)
            trades = list(self._trades.get(product, []))

        if len(trades) < 2:
            return None

        prices = np.array([t.price for t in trades])
        volumes = np.array([t.volume for t in trades])
        timestamps = np.array([t.timestamp for t in trades])

        n = len(prices)
        total_vol = volumes.sum()

        # --- VWAP ---
        vwap = np.average(prices, weights=volumes) if total_vol > 0 else np.mean(prices)

        # --- Exponentially-weighted mean ---
        # Half-life = 1/4 of window (recent trades 4x more important)
        now = time.time()
        ages = now - timestamps  # seconds ago
        half_life = self.window / 4
        decay_weights = np.exp(-ages * np.log(2) / half_life)
        combined_weights = decay_weights * volumes  # weight by both recency AND volume
        if combined_weights.sum() > 0:
            ew_mean = np.average(prices, weights=combined_weights)
            # Weighted std dev
            ew_var = np.average((prices - ew_mean) ** 2, weights=combined_weights)
            ew_std = max(1.0, np.sqrt(ew_var))
        else:
            ew_mean = np.mean(prices)
            ew_std = max(1.0, np.std(prices))

        # --- Gaussian fit ---
        gauss_mean = np.mean(prices)
        gauss_std = max(1.0, np.std(prices))

        # If enough trades, try proper Gaussian MLE
        if n >= 5:
            try:
                gauss_mean, gauss_std = norm.fit(prices)
                gauss_std = max(1.0, gauss_std)
            except Exception:
                pass

        return {
            "vwap": float(vwap),
            "ew_mean": float(ew_mean),
            "ew_std": float(ew_std),
            "gauss_mean": float(gauss_mean),
            "gauss_std": float(gauss_std),
            "n_trades": n,
            "total_volume": int(total_vol),
        }

    def get_consensus(self, product, cfg: Config) -> tuple[float | None, float | None]:
        """Returns (market_consensus_price, market_uncertainty).

        consensus = weighted blend of VWAP + EW mean + Gaussian mean
        uncertainty = blend of std devs (how noisy/disagreeable the market is)
        """
        stats = self.get_stats(product)
        if stats is None:
            return None, None

        consensus = (
            cfg.VWAP_WEIGHT * stats["vwap"] +
            cfg.EW_MEAN_WEIGHT * stats["ew_mean"] +
            cfg.GAUSSIAN_MEAN_WEIGHT * stats["gauss_mean"]
        )

        # Uncertainty is the average of our std dev measures
        uncertainty = (stats["ew_std"] + stats["gauss_std"]) / 2

        return consensus, uncertainty


# ============================================================
# 2. EDGE CALCULATOR — combines theo with market data
# ============================================================
class EdgeCalculator:
    """Computes trading edge by comparing our theo to market consensus.

    EDGE CALCULATION:
    ┌──────────────────────────────────────────────────────┐
    │ edge = theo - market_consensus                       │
    │   positive edge → market is BELOW our theo → BUY     │
    │   negative edge → market is ABOVE our theo → SELL    │
    │                                                      │
    │ edge_zscore = edge / market_uncertainty               │
    │   How many "standard deviations" of market noise     │
    │   our edge represents. A zscore of 2+ means the     │
    │   market is trading 2σ away from our fair value —    │
    │   that's statistically significant.                  │
    │                                                      │
    │ adjusted_edge = edge_zscore * confidence              │
    │   Scaled by how much we trust our model.             │
    │   High confidence + high zscore = strong signal.     │
    │   Low confidence + low zscore = noise, don't trade.  │
    └──────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Config):
        self.cfg = config

    def compute_edge(self, product, theo, market_consensus, market_uncertainty):
        """Returns an EdgeInfo with all edge metrics."""
        if theo is None or market_consensus is None:
            return None

        raw_edge = theo - market_consensus  # positive = market undervalued
        edge_pct = raw_edge / max(abs(theo), 1)

        # Z-score: how many market-std-devs away is our theo?
        if market_uncertainty and market_uncertainty > 0:
            edge_zscore = raw_edge / market_uncertainty
        else:
            edge_zscore = raw_edge / max(abs(theo) * 0.01, 1)

        confidence = self.cfg.PRODUCT_CONFIDENCE.get(product, 0.5)
        adjusted_edge = edge_zscore * confidence

        return {
            "raw_edge": raw_edge,           # theo - market (in price units)
            "edge_pct": edge_pct,           # as fraction of theo
            "edge_zscore": edge_zscore,     # in units of market σ
            "adjusted_edge": adjusted_edge, # zscore × confidence
            "confidence": confidence,
            "market_consensus": market_consensus,
            "market_uncertainty": market_uncertainty,
            "theo": theo,
            "direction": "BUY" if raw_edge > 0 else "SELL",
        }


# ============================================================
# 3. EDGE-BASED SIZER
# ============================================================
class EdgeBasedSizer:
    """Determines position sizing from edge metrics.

    SIZING RULES:
    ┌────────────────────────────────────────────────────────┐
    │ |adjusted_edge| < 0.3  → don't trade (noise)          │
    │ |adjusted_edge| 0.3-1  → size 1-2 (weak signal)       │
    │ |adjusted_edge| 1-2    → size 2-4 (moderate signal)    │
    │ |adjusted_edge| 2-3    → size 4-7 (strong signal)      │
    │ |adjusted_edge| > 3    → size up to max (screaming)    │
    │                                                        │
    │ Then scale by position room remaining.                 │
    └────────────────────────────────────────────────────────┘

    The adjusted_edge combines:
    - How far the market is from our theo (in σ units)
    - How confident we are in our model
    So a zscore of 2 with confidence 0.8 → adjusted = 1.6 → moderate
    A zscore of 2 with confidence 0.3 → adjusted = 0.6 → weak
    """

    def __init__(self, config: Config):
        self.cfg = config

    def size_from_edge(self, edge_info, current_position, product=None) -> int:
        """Compute order size from edge metrics."""
        if edge_info is None:
            return self.cfg.MIN_ORDER_SIZE

        ae = abs(edge_info["adjusted_edge"])

        # Map adjusted_edge to raw size — much more aggressive than v3
        if ae < 0.3:
            return 0  # noise — don't trade
        elif ae < 1.0:
            raw_size = 1 + (ae - 0.3) / 0.7 * 2    # 1 to 3
        elif ae < 2.0:
            raw_size = 3 + (ae - 1.0) / 1.0 * 4    # 3 to 7
        elif ae < 3.0:
            raw_size = 7 + (ae - 2.0) / 1.0 * 5    # 7 to 12
        else:
            raw_size = self.cfg.MAX_ORDER_SIZE       # 15

        # Group A ×4 scoring multiplier → trade more aggressively
        if product and product in self.cfg.GROUP_A:
            raw_size *= self.cfg.GROUP_A_SIZE_MULT

        # Position room scaling — smoother ramp-down near limits
        is_buying = edge_info["direction"] == "BUY"
        if is_buying:
            room = self.cfg.MAX_POSITION - current_position
        else:
            room = self.cfg.MAX_POSITION + current_position

        room_factor = min(1.0, max(0.0, room / 30))
        raw_size *= room_factor

        return int(max(0, min(self.cfg.MAX_ORDER_SIZE, round(raw_size))))

    def size_for_quote(self, edge_info, current_position, product=None) -> int:
        """Size for passive resting orders."""
        if edge_info is None:
            confidence = 0.5
        else:
            confidence = edge_info["confidence"]

        position_room = max(0.0, 1.0 - abs(current_position) / self.cfg.MAX_POSITION)
        raw = self.cfg.BASE_ORDER_SIZE * confidence * position_room

        # Group A gets more size
        if product and product in self.cfg.GROUP_A:
            raw *= self.cfg.GROUP_A_SIZE_MULT

        return int(max(self.cfg.MIN_ORDER_SIZE, min(self.cfg.MAX_ORDER_SIZE, round(raw))))

    def size_for_arb(self, edge_points, positions_dict, legs):
        """Size for multi-leg arb — risk-free profit, be aggressive."""
        if edge_points < self.cfg.ARB_MIN_PROFIT:
            return 0

        # Much more aggressive arb sizing — arb is risk-free
        if edge_points < 5:
            raw = 3
        elif edge_points < 15:
            raw = 5
        elif edge_points < 50:
            raw = 8
        else:
            raw = self.cfg.ARB_MAX_SIZE

        for product, direction in legs.items():
            pos = positions_dict.get(product, 0)
            room = (self.cfg.MAX_POSITION - pos) if direction == "BUY" else (self.cfg.MAX_POSITION + pos)
            raw = min(raw, max(0, room))

        return int(raw)


# ============================================================
# 4. ALPHA ENGINE (same as v2, included for completeness)
# ============================================================
class AlphaEngine:
    def __init__(self, config: Config):
        self.cfg = config
        self.london_lat, self.london_lon = 51.5074, -0.1278
        self.start_time = pd.to_datetime(config.START_TIME).tz_localize("Europe/London")
        self.settlement_time = pd.to_datetime(config.SETTLEMENT_TIME).tz_localize("Europe/London")
        self._weather_df = None
        self._tide_df = None
        self._tide_model_params = None

    def fetch_weather(self):
        try:
            resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
                "latitude": self.london_lat, "longitude": self.london_lon,
                "minutely_15": "temperature_2m,relative_humidity_2m,wind_speed_10m,visibility",
                "past_minutely_15": 96, "forecast_minutely_15": 96, "timezone": "Europe/London",
            }, timeout=10)
            if not resp.ok: return None
            data = resp.json()["minutely_15"]
            df = pd.DataFrame({
                "time": pd.to_datetime(data["time"]).tz_localize("Europe/London"),
                "temp_c": data["temperature_2m"],
                "humidity": data["relative_humidity_2m"],
                "wind_kph": data["wind_speed_10m"],
                "visibility_m": data["visibility"],
            })
            df["temp_f"] = df["temp_c"] * 9 / 5 + 32
            df["temp_f_rounded"] = df["temp_f"].round()
            df["wx_value"] = df["temp_f_rounded"] * df["humidity"]
            self._weather_df = df
            return df
        except Exception as e:
            print(f"[Weather] Error: {e}")
            return None

    def calc_wx_spot(self):
        df = self._weather_df
        if df is None: return None
        rows = df[df["time"] == self.settlement_time]
        if not rows.empty: return float(rows["wx_value"].iloc[0])
        window = df[(df["time"] >= self.start_time) & (df["time"] <= self.settlement_time)]
        return float(window["wx_value"].iloc[-1]) if not window.empty else None

    def calc_wx_sum(self):
        df = self._weather_df
        if df is None: return None
        window = df[(df["time"] >= self.start_time) & (df["time"] <= self.settlement_time)]
        return float((window["wx_value"] / 100).sum()) if not window.empty else None

    def fetch_tides(self):
        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://environment.data.gov.uk/flood-monitoring/id/measures/"
                    "0006-level-tidal_level-i-15_min-mAOD/readings",
                    params={"_sorted": "", "_limit": 400}, timeout=30)
                if not resp.ok: return None
                items = resp.json().get("items", [])
                df = pd.DataFrame(items)[["dateTime", "value"]]
                df["time"] = pd.to_datetime(df["dateTime"], utc=True).dt.tz_convert("Europe/London")
                df["level"] = df["value"].astype(float)
                df = df.sort_values("time").dropna(subset=["level"]).reset_index(drop=True)
                self._tide_df = df
                self._fit_tidal_model(df)
                return df
            except Exception as e:
                print(f"[Tides] Attempt {attempt+1} error: {e}")
                if attempt < 2:
                    time.sleep(5)
        return None

    def _fit_tidal_model(self, df):
        try:
            t_hours = (df["time"] - df["time"].min()).dt.total_seconds() / 3600.0
            # [CHANGE] Dual harmonic wave function (M2 + S2 tides)
            def tidal_wave(t, a1, omega1, phi1, a2, omega2, phi2, c):
                return a1 * np.sin(omega1 * t + phi1) + a2 * np.sin(omega2 * t + phi2) + c
            # [CHANGE] Initial guesses for 12.42h (Lunar) and 12.00h (Solar) cycles
            p0 = [3.0, 2 * np.pi / 12.42, 0.0, 1.0, 2 * np.pi / 12.00, 0.0, 0.0]
            popt, _ = curve_fit(tidal_wave, t_hours.values, df["level"].values, p0=p0, maxfev=5000)
            self._tide_model_params = popt
        except Exception as e:
            print(f"[Tides] Fit error: {e}")
            self._tide_model_params = None

    def _predict_tide(self, target_time):
        if self._tide_df is None or self._tide_model_params is None: return None
        # [CHANGE] Dual harmonic wave function
        def tidal_wave(t, a1, omega1, phi1, a2, omega2, phi2, c):
            return a1 * np.sin(omega1 * t + phi1) + a2 * np.sin(omega2 * t + phi2) + c
        t0 = self._tide_df["time"].min()
        h = (target_time - t0).total_seconds() / 3600.0
        return float(tidal_wave(h, *self._tide_model_params))

    def calc_tide_spot(self):
        prediction = self._predict_tide(self.settlement_time)
        if prediction is None: return None
        surge = 0.0
        if self._weather_df is not None:
            rows = self._weather_df[self._weather_df["time"] == self.settlement_time]
            if not rows.empty:
                wind = rows["wind_kph"].iloc[0]
                if wind > 25: surge = (wind - 25) * 0.02
        return abs(prediction + surge) * 1000

    def calc_tide_swing(self):
        if self._tide_df is None: return None
        def strangle(d):
            return max(0.0, 20.0 - d) + max(0.0, d - 25.0) if not pd.isna(d) else 0.0
        now = pd.Timestamp.now(tz="Europe/London")
        realized = self._tide_df[(self._tide_df["time"] >= self.start_time) & (self._tide_df["time"] <= now)].copy()
        realized_sum = 0.0
        if len(realized) > 1:
            realized_sum = float(realized["level"].diff().abs().mul(100).apply(strangle).sum())
        forecast_sum = 0.0
        if self._tide_model_params is not None:
            # [CHANGE] Dual harmonic wave function
            def tidal_wave(t, a1, omega1, phi1, a2, omega2, phi2, c):
                return a1 * np.sin(omega1 * t + phi1) + a2 * np.sin(omega2 * t + phi2) + c
            t0 = self._tide_df["time"].min()
            future = pd.date_range(start=now, end=self.settlement_time, freq="15min")
            if len(future) > 1:
                fh = (future - t0).total_seconds() / 3600.0
                fl = tidal_wave(fh.values, *self._tide_model_params)
                forecast_sum = sum(strangle(d) for d in np.abs(np.diff(fl)) * 100)
        return realized_sum + forecast_sum

    def fetch_and_calc_flights(self):
        """Markets 5 & 6: LHR flight count and imbalance index.

        Fixes over naive approach:
        1. Codeshare deduplication — same physical plane appears multiple times
           under different airline codes. We deduplicate by scheduled time + 
           departure airport (arrivals) or destination (departures).
        2. Hourly flight profile — Heathrow has a night ban (roughly 23:00-05:30).
           Instead of flat extrapolation, we use an hourly activity profile.
        3. Cancelled flight filtering — only count flights with actual movement.
        """
        if not self.cfg.RAPID_API_KEY or self.cfg.RAPID_API_KEY == "YOUR_RAPID_API_KEY":
            return None, None

        now = pd.Timestamp.now(tz="Europe/London")
        elapsed = max(1, int((now - self.start_time).total_seconds() / 60))
        fetch = min(elapsed, 720)

        try:
            resp = requests.get(
                "https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR",
                headers={"x-rapidapi-host": "aerodatabox.p.rapidapi.com",
                         "x-rapidapi-key": self.cfg.RAPID_API_KEY},
                params={"offsetMinutes": -fetch, "durationMinutes": fetch, "direction": "Both"},
                timeout=15)
            if not resp.ok: return None, None
            data = resp.json()

            raw_arrivals = data.get("arrivals", [])
            raw_departures = data.get("departures", [])

            # --- Deduplicate codeshares ---
            arrivals = self._deduplicate_flights(raw_arrivals, "arrival")
            departures = self._deduplicate_flights(raw_departures, "departure")

            realized_arr = len(arrivals)
            realized_dep = len(departures)
            realized_total = realized_arr + realized_dep

            print(f"[Flights] Raw: {len(raw_arrivals)}arr + {len(raw_departures)}dep = {len(raw_arrivals)+len(raw_departures)} | "
                  f"Deduped: {realized_arr}arr + {realized_dep}dep = {realized_total}")

            # --- Forecast remaining flights using hourly profile ---
            remaining_minutes = max(1, int((self.settlement_time - now).total_seconds() / 60))

            # Heathrow typical hourly flight profile (fraction of daily total per hour)
            # Based on public Heathrow stats: ~1300 flights/day
            # Night ban roughly 23:00-05:30, peak 06:00-22:00
            HOURLY_PROFILE = {
                0: 0.010, 1: 0.005, 2: 0.003, 3: 0.002, 4: 0.005, 5: 0.020,
                6: 0.060, 7: 0.075, 8: 0.080, 9: 0.075, 10: 0.070, 11: 0.065,
                12: 0.065, 13: 0.065, 14: 0.065, 15: 0.060, 16: 0.060, 17: 0.055,
                18: 0.055, 19: 0.050, 20: 0.045, 21: 0.040, 22: 0.030, 23: 0.020,
            }
            # Typical total daily flights at Heathrow (both directions)
            TYPICAL_DAILY_FLIGHTS = 1300

            # Calculate how many flights we expect in the remaining window
            # by summing the hourly profile fractions for each future hour
            forecast_flights = 0
            cursor = now.ceil("h")  # round up to next hour
            while cursor <= self.settlement_time:
                hour = cursor.hour
                fraction = HOURLY_PROFILE.get(hour, 0.04)  # default 4%

                # How much of this hour falls in our remaining window?
                hour_start = cursor
                hour_end = cursor + pd.Timedelta(hours=1)
                overlap_start = max(hour_start, now)
                overlap_end = min(hour_end, self.settlement_time)
                overlap_frac = max(0, (overlap_end - overlap_start).total_seconds() / 3600)

                forecast_flights += TYPICAL_DAILY_FLIGHTS * fraction * overlap_frac
                cursor += pd.Timedelta(hours=1)

            # Weather dampening on the forecast
            dampener = 1.0
            if self._weather_df is not None:
                fw = self._weather_df[self._weather_df["time"] > now]
                if len(fw) > 0:
                    poor_vis_frac = len(fw[fw["visibility_m"] < 2000]) / len(fw)
                    high_wind_frac = len(fw[fw["wind_kph"] > 40]) / len(fw)
                    dampener = max(0.5, 1.0 - poor_vis_frac * 0.3 - high_wind_frac * 0.2)

            lhr_count = realized_total + forecast_flights * dampener

            # --- LHR_INDEX: imbalance per 30-min interval ---
            # The settlement formula sums |100*(arr-dep)/(arr+dep)| per 30-min interval
            # For realized data, compute properly from the flight timestamps
            # For forecast, assume roughly balanced (imbalance averages out)
            realized_imbalance = self._calc_flight_imbalance(arrivals, departures, now)

            # Forecast remaining intervals assuming slight imbalance
            remaining_intervals = remaining_minutes / 30
            # Historical Heathrow imbalance ~2-4 per interval (many overnight intervals are 0)
            avg_imbalance_per_interval = 2.5
            forecast_imbalance = remaining_intervals * avg_imbalance_per_interval

            lhr_index = realized_imbalance + forecast_imbalance

            return lhr_count, lhr_index
        except Exception as e:
            print(f"[Flights] Error: {e}")
            return None, None

    def _deduplicate_flights(self, flights, direction):
        """Remove codeshare duplicates from flight list.

        Strategy: Group by (scheduled_time, origin/destination airport).
        Codeshares share the same physical flight, so they have the same
        scheduled time and route. We keep only unique physical flights.

        Also filters out cancelled flights where possible.
        """
        seen = set()
        unique = []

        for f in flights:
            # Skip cancelled flights
            status = f.get("status", "")
            if isinstance(status, str) and "cancel" in status.lower():
                continue

            # Build a dedup key from the flight's physical characteristics
            # Use scheduled time + origin/destination as the unique identifier
            sched_time = ""
            if direction == "arrival":
                # For arrivals: use departure time + departure airport
                departure = f.get("departure", {})
                sched_time = departure.get("scheduledTime", {}).get("utc", "")
                airport = departure.get("airport", {}).get("iata", "")
            else:
                # For departures: use arrival time + arrival airport
                arrival = f.get("arrival", {})
                sched_time = arrival.get("scheduledTime", {}).get("utc", "")
                airport = arrival.get("airport", {}).get("iata", "")

            # Fallback: use the movement time at LHR
            if not sched_time:
                movement = f.get("arrival" if direction == "arrival" else "departure", {})
                sched_time = movement.get("scheduledTime", {}).get("utc", "")
                # Use flight number as fallback key component
                airport = f.get("number", "")

            key = f"{sched_time}_{airport}"

            if key and key not in seen:
                seen.add(key)
                unique.append(f)
            elif not key:
                # If we can't build a key, include it anyway (don't lose data)
                unique.append(f)

        return unique

    def _calc_flight_imbalance(self, arrivals, departures, now):
        """Calculate realized LHR_INDEX from flight data.

        Groups flights into 30-min intervals and computes:
        sum of |100 * (arr - dep) / (arr + dep)| per interval
        """
        try:
            # Extract timestamps from flights
            def get_time(flight, direction):
                key = "arrival" if direction == "arrival" else "departure"
                movement = flight.get(key, {})
                time_info = movement.get("actualTime", movement.get("scheduledTime", {}))
                if isinstance(time_info, dict):
                    t = time_info.get("local", time_info.get("utc", ""))
                else:
                    t = str(time_info)
                try:
                    return pd.to_datetime(t)
                except Exception:
                    return None

            # Count arrivals and departures per 30-min interval
            interval_counts = {}  # interval_key -> {"arr": 0, "dep": 0}

            for f in arrivals:
                t = get_time(f, "arrival")
                if t is not None:
                    key = t.floor("30min")
                    if key not in interval_counts:
                        interval_counts[key] = {"arr": 0, "dep": 0}
                    interval_counts[key]["arr"] += 1

            for f in departures:
                t = get_time(f, "departure")
                if t is not None:
                    key = t.floor("30min")
                    if key not in interval_counts:
                        interval_counts[key] = {"arr": 0, "dep": 0}
                    interval_counts[key]["dep"] += 1

            # Sum imbalance metric across intervals
            total_imbalance = 0.0
            for key, counts in interval_counts.items():
                a, d = counts["arr"], counts["dep"]
                if a + d > 0:
                    total_imbalance += abs(100 * (a - d) / (a + d))

            return total_imbalance
        except Exception as e:
            print(f"[Flights] Imbalance calc error: {e}")
            return 0.0

    def calc_etf(self, t, w, l):
        return (t + w + l) if all(v is not None for v in [t, w, l]) else None

    def calc_options(self, S):
        if S is None: return None
        return 2*max(0, 6200-S) + max(0, S-6200) - 2*max(0, S-6600) + 3*max(0, S-7000)

    def compute_all_theos(self):
        theos = {}
        for key, fn in [("WX_SPOT", self.calc_wx_spot), ("WX_SUM", self.calc_wx_sum),
                        ("TIDE_SPOT", self.calc_tide_spot), ("TIDE_SWING", self.calc_tide_swing)]:
            v = fn()
            if v is not None: theos[key] = v
        return theos


# ============================================================
# 5. UNCERTAINTY MODEL
# ============================================================
class UncertaintyModel:
    def __init__(self, start_time, settlement_time, base_width_pct, min_width):
        self.start_time = start_time
        self.settlement_time = settlement_time
        self.base_width_pct = base_width_pct
        self.min_width = min_width
        self.total_seconds = (settlement_time - start_time).total_seconds()

    def get_fraction_elapsed(self):
        now = pd.Timestamp.now(tz="Europe/London")
        return max(0.0, min(1.0, (now - self.start_time).total_seconds() / self.total_seconds))

    def get_uncertainty_factor(self, product=None):
        frac = self.get_fraction_elapsed()
        if product in {"WX_SUM", "TIDE_SWING", "LHR_COUNT", "LHR_INDEX"}:
            return max(0.1, 1.0 - 0.9 * frac)
        return max(0.2, 1.0 - 0.8 * math.sqrt(frac))

    def get_half_width(self, theo, product=None, market_std=None):
        """Width from our model uncertainty, optionally blended with market σ."""
        model_width = max(self.min_width, theo * self.base_width_pct * self.get_uncertainty_factor(product))

        # If we have market std dev data, blend it in
        # Market σ tells us how noisy the market is — quote wider in noisy markets
        if market_std is not None and market_std > 0:
            market_width = max(self.min_width, market_std * 1.5)
            # Blend: 60% our model, 40% market observation
            return 0.6 * model_width + 0.4 * market_width
        return model_width


# ============================================================
# 6. RATE LIMITER
# ============================================================
class RateLimiter:
    def __init__(self, min_interval):
        self.min_interval = min_interval
        self._last = {}
        self._lock = Lock()

    def can_act(self, key):
        with self._lock:
            return (time.monotonic() - self._last.get(key, 0)) >= self.min_interval

    def mark(self, key):
        with self._lock:
            self._last[key] = time.monotonic()


# ============================================================
# 7. THE TRADING BOT
# ============================================================
class ImprovedTradingBot(BaseBot):

    def __init__(self, config: Config):
        super().__init__(config.EXCHANGE_URL, config.USERNAME, config.PASSWORD)
        self.cfg = config
        self.alpha = AlphaEngine(config)

        self.theos = {}
        self._prev_theos = {}
        self.latest_books = {}
        self._lock = Lock()

        start = pd.to_datetime(config.START_TIME).tz_localize("Europe/London")
        settle = pd.to_datetime(config.SETTLEMENT_TIME).tz_localize("Europe/London")

        self.trade_tracker = TradeTracker(window_seconds=config.TRADE_HISTORY_WINDOW)
        self.edge_calc = EdgeCalculator(config)
        self.sizer = EdgeBasedSizer(config)
        self.uncertainty = UncertaintyModel(start, settle, config.BASE_QUOTE_WIDTH_PCT, config.MIN_QUOTE_WIDTH)
        self.quote_limiter = RateLimiter(config.MIN_SECONDS_BETWEEN_REQUOTES)
        self.arb_limiter = RateLimiter(1.0)

    # ----------------------------------------------------------
    # Background tasks
    # ----------------------------------------------------------
    def start_background_tasks(self):
        def data_loop():
            while True:
                try:
                    self.alpha.fetch_weather()
                    self.alpha.fetch_tides()
                    new = self.alpha.compute_all_theos()
                    with self._lock:
                        self.theos.update(new)
                        self._update_derived()
                    self._log_theos("Data")
                except Exception as e:
                    print(f"[DataLoop] {e}")
                time.sleep(self.cfg.WEATHER_REFRESH_SECS)

        def flight_loop():
            while True:
                try:
                    c, i = self.alpha.fetch_and_calc_flights()
                    with self._lock:
                        if c is not None: self.theos["LHR_COUNT"] = c
                        if i is not None: self.theos["LHR_INDEX"] = i
                        self._update_derived()
                    self._log_theos("Flights")
                except Exception as e:
                    print(f"[FlightLoop] {e}")
                time.sleep(self.cfg.FLIGHT_REFRESH_SECS)

        def market_trade_poll():
            """Periodically poll market trades to fill trade tracker."""
            while True:
                try:
                    trades = self.get_market_trades()
                    if trades:
                        self.trade_tracker.add_market_trades(trades)
                except Exception as e:
                    print(f"[TradePoll] {e}")
                time.sleep(self.cfg.MARKET_TRADE_POLL_SECS)

        Thread(target=data_loop, daemon=True).start()
        Thread(target=flight_loop, daemon=True).start()
        Thread(target=market_trade_poll, daemon=True).start()
        print("[Bot] All background tasks started.")

    def _update_derived(self):
        t, w, l = self.theos.get("TIDE_SPOT"), self.theos.get("WX_SPOT"), self.theos.get("LHR_COUNT")
        etf = self.alpha.calc_etf(t, w, l)
        if etf is not None:
            self.theos["LON_ETF"] = etf
            opt = self.alpha.calc_options(etf)
            if opt is not None:
                self.theos["LON_FLY"] = opt

    def _log_theos(self, src):
        with self._lock:
            r = {k: round(v, 1) for k, v in self.theos.items()}
        print(f"[Theos][{src}] {r}")

    # ----------------------------------------------------------
    # SSE Callbacks
    # ----------------------------------------------------------
    def on_trades(self, trade):
        # Track ALL trades for market statistics
        self.trade_tracker.add_trade(trade.product, trade.price, trade.volume)

        # Only log OUR fills (SSE streams every trade in the market)
        if trade.buyer == self.username:
            print(f"[FILL] BOUGHT {trade.volume}x {trade.product} @ {trade.price}")
        elif trade.seller == self.username:
            print(f"[FILL] SOLD {trade.volume}x {trade.product} @ {trade.price}")

    def on_orderbook(self, ob):
        self.latest_books[ob.product] = ob

        if ob.product in ("LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"):
            self._check_etf_arb()
        if ob.product == "LON_FLY":
            self._check_options_arb()

        self._maybe_requote(ob.product)

    # ----------------------------------------------------------
    # Market Making with Full Edge Pipeline
    # ----------------------------------------------------------
    def _maybe_requote(self, product):
        if product not in self.theos:
            return
        if not self.quote_limiter.can_act(f"quote_{product}"):
            return

        theo = self.theos[product]
        prev = self._prev_theos.get(product)
        if prev is not None:
            if abs(theo - prev) / max(abs(prev), 1) < self.cfg.THEO_CHANGE_THRESHOLD_PCT:
                return

        self.quote_limiter.mark(f"quote_{product}")
        self._prev_theos[product] = theo
        self._place_quotes(product, theo)

    def _place_quotes(self, product, theo):
        try:
            positions = self.get_positions()
        except Exception:
            return
        pos = positions.get(product, 0)

        # Get market stats for this product
        consensus, market_std = self.trade_tracker.get_consensus(product, self.cfg)

        # ── Theo-market blending ──
        # For low-confidence products, blend theo toward market consensus
        # This prevents trading aggressively against the market when our model is weak
        effective_theo = theo
        confidence = self.cfg.PRODUCT_CONFIDENCE.get(product, 0.5)
        if self.cfg.BLEND_THEO_WITH_MARKET and consensus is not None:
            effective_theo = confidence * theo + (1 - confidence) * consensus

        # Compute edge info using effective (blended) theo
        edge_info = None
        if consensus is not None:
            edge_info = self.edge_calc.compute_edge(product, effective_theo, consensus, market_std)
            if edge_info:
                ae = edge_info["adjusted_edge"]
                direction = edge_info["direction"]
                if abs(ae) > 0.5:
                    blend_tag = f" blend={effective_theo:.0f}" if effective_theo != theo else ""
                    print(f"[Edge] {product}: theo={theo:.0f}{blend_tag} mkt={consensus:.0f} "
                          f"σ={market_std:.1f} z={edge_info['edge_zscore']:.2f} "
                          f"ae={ae:.2f} → {direction}")

        # ── Position-aware skew ──
        skew = (pos / self.cfg.MAX_POSITION) * self.cfg.SKEW_FACTOR
        skewed_theo = effective_theo - skew

        # Strong edge: shift quoting center toward our view
        if edge_info and abs(edge_info["adjusted_edge"]) > 1.0:
            edge_shift = edge_info["raw_edge"] * 0.25
            skewed_theo += edge_shift

        # ── Dynamic width ──
        half_width = self.uncertainty.get_half_width(effective_theo, product, market_std)
        wide_half = half_width * 2

        # ── Time-based aggression ──
        # As settlement approaches and uncertainty drops, increase sizes
        frac_elapsed = self.uncertainty.get_fraction_elapsed()
        time_boost = 1.0
        if frac_elapsed > 0.3:
            time_boost = 1.0 + (frac_elapsed - 0.3) * 1.5  # up to 2.05× at settlement

        # ── Sizing ──
        tight_size = self.sizer.size_for_quote(edge_info, pos, product)
        tight_size = int(min(self.cfg.MAX_ORDER_SIZE, tight_size * time_boost))
        tight_size = max(self.cfg.MIN_ORDER_SIZE, tight_size)
        wide_size = max(1, tight_size + 2)
        wide_size = min(self.cfg.MAX_ORDER_SIZE, wide_size)

    # ── Price levels ──
        tb = max(1, math.floor(skewed_theo - half_width))
        ta = math.ceil(skewed_theo + half_width)
        wb = max(1, math.floor(skewed_theo - wide_half))
        wa = math.ceil(skewed_theo + wide_half)
        
        # [NEW] Deep book "lottery" quotes to catch spikes
        lottery_half = half_width * self.cfg.LOTTERY_WIDTH_MULT
        lb = max(1, math.floor(skewed_theo - lottery_half))
        la = math.ceil(skewed_theo + lottery_half)

        if ta <= tb: ta = tb + 1
        if wa <= wb: wa = wb + 1
        if la <= wa: la = wa + 1
        if lb >= wb: lb = wb - 1

        # Cancel old orders first
        self._cancel_product_orders(product)

# Cancel old orders first
        self._cancel_product_orders(product)

        orders = []
        if pos < self.cfg.MAX_POSITION:
            orders.append(OrderRequest(product, tb, Side.BUY, tight_size))
            if wb < tb:
                orders.append(OrderRequest(product, wb, Side.BUY, wide_size))
            if lb < wb:
                orders.append(OrderRequest(product, lb, Side.BUY, self.cfg.LOTTERY_ORDER_SIZE))

        if pos > -self.cfg.MAX_POSITION:
            orders.append(OrderRequest(product, ta, Side.SELL, tight_size))
            if wa > ta:
                orders.append(OrderRequest(product, wa, Side.SELL, wide_size))
            if la > wa:
                orders.append(OrderRequest(product, la, Side.SELL, self.cfg.LOTTERY_ORDER_SIZE))

        # ── Aggressive directional orders when edge is large ──
        # Cross the spread when our model strongly disagrees with market
        if edge_info and abs(edge_info["adjusted_edge"]) > 1.5:
            agg_size = self.sizer.size_from_edge(edge_info, pos, product)
            agg_size = int(min(self.cfg.MAX_ORDER_SIZE, agg_size * time_boost))
            if agg_size > 0:
                if edge_info["direction"] == "BUY" and pos < self.cfg.MAX_POSITION:
                    # Aggressive bid closer to the ask
                    agg_price = math.ceil(skewed_theo - half_width * 0.3)
                    orders.append(OrderRequest(product, max(1, agg_price), Side.BUY, agg_size))
                elif edge_info["direction"] == "SELL" and pos > -self.cfg.MAX_POSITION:
                    agg_price = math.floor(skewed_theo + half_width * 0.3)
                    orders.append(OrderRequest(product, agg_price, Side.SELL, agg_size))

        if orders:
            self.send_orders(orders)

    def _cancel_product_orders(self, product):
        try:
            for o in self.get_orders(product):
                self.cancel_order(o["id"])
        except Exception:
            pass

    # ----------------------------------------------------------
    # ETF Arbitrage (edge-scaled)
    # ----------------------------------------------------------
    def _check_etf_arb(self):
        req = ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        if not all(p in self.latest_books for p in req):
            return
        if not self.arb_limiter.can_act("etf_arb"):
            return

        eb = self.latest_books["LON_ETF"]
        tb = self.latest_books["TIDE_SPOT"]
        wb = self.latest_books["WX_SPOT"]
        lb = self.latest_books["LHR_COUNT"]

        try:
            positions = self.get_positions()
        except Exception:
            return

        # Direction 1: Sell ETF, Buy Components
        if eb.buy_orders and tb.sell_orders and wb.sell_orders and lb.sell_orders:
            edge = eb.buy_orders[0].price - (tb.sell_orders[0].price + wb.sell_orders[0].price + lb.sell_orders[0].price)
            if edge > self.cfg.ARB_MIN_PROFIT:
                legs = {"LON_ETF": "SELL", "TIDE_SPOT": "BUY", "WX_SPOT": "BUY", "LHR_COUNT": "BUY"}
                sz = self.sizer.size_for_arb(edge, positions, legs)
                if sz > 0:
                    self.arb_limiter.mark("etf_arb")
                    print(f"[ARB] SELL ETF, BUY components. edge={edge:.0f} size={sz}")
                    self._send_ioc(OrderRequest("LON_ETF", eb.buy_orders[0].price, Side.SELL, sz))
                    self._send_ioc(OrderRequest("TIDE_SPOT", tb.sell_orders[0].price, Side.BUY, sz))
                    self._send_ioc(OrderRequest("WX_SPOT", wb.sell_orders[0].price, Side.BUY, sz))
                    self._send_ioc(OrderRequest("LHR_COUNT", lb.sell_orders[0].price, Side.BUY, sz))

        # Direction 2: Buy ETF, Sell Components
        if eb.sell_orders and tb.buy_orders and wb.buy_orders and lb.buy_orders:
            edge = (tb.buy_orders[0].price + wb.buy_orders[0].price + lb.buy_orders[0].price) - eb.sell_orders[0].price
            if edge > self.cfg.ARB_MIN_PROFIT:
                legs = {"LON_ETF": "BUY", "TIDE_SPOT": "SELL", "WX_SPOT": "SELL", "LHR_COUNT": "SELL"}
                sz = self.sizer.size_for_arb(edge, positions, legs)
                if sz > 0:
                    self.arb_limiter.mark("etf_arb")
                    print(f"[ARB] BUY ETF, SELL components. edge={edge:.0f} size={sz}")
                    self._send_ioc(OrderRequest("LON_ETF", eb.sell_orders[0].price, Side.BUY, sz))
                    self._send_ioc(OrderRequest("TIDE_SPOT", tb.buy_orders[0].price, Side.SELL, sz))
                    self._send_ioc(OrderRequest("WX_SPOT", wb.buy_orders[0].price, Side.SELL, sz))
                    self._send_ioc(OrderRequest("LHR_COUNT", lb.buy_orders[0].price, Side.SELL, sz))

    # ----------------------------------------------------------
    # Options Arbitrage (edge-scaled)
    # ----------------------------------------------------------
    def _check_options_arb(self):
        if "LON_FLY" not in self.latest_books or "LON_FLY" not in self.theos:
            return
        if not self.arb_limiter.can_act("options_arb"):
            return

        book = self.latest_books["LON_FLY"]
        theo = self.theos["LON_FLY"]

        try:
            pos = self.get_positions().get("LON_FLY", 0)
        except Exception:
            return

        consensus, mkt_std = self.trade_tracker.get_consensus("LON_FLY", self.cfg)

        if book.sell_orders:
            ask = book.sell_orders[0].price
            edge_info = self.edge_calc.compute_edge("LON_FLY", theo, ask, mkt_std or abs(theo)*0.02)
            sz = self.sizer.size_from_edge(edge_info, pos) if edge_info else 0
            if sz > 0 and ask < theo - self.cfg.ARB_MIN_PROFIT:
                self.arb_limiter.mark("options_arb")
                print(f"[ARB] BUY LON_FLY @{ask} theo={theo:.0f} size={sz}")
                self._send_ioc(OrderRequest("LON_FLY", ask, Side.BUY, sz))

        if book.buy_orders:
            bid = book.buy_orders[0].price
            edge_info = self.edge_calc.compute_edge("LON_FLY", theo, bid, mkt_std or abs(theo)*0.02)
            sz = self.sizer.size_from_edge(edge_info, pos) if edge_info else 0
            if sz > 0 and bid > theo + self.cfg.ARB_MIN_PROFIT:
                self.arb_limiter.mark("options_arb")
                print(f"[ARB] SELL LON_FLY @{bid} theo={theo:.0f} size={sz}")
                self._send_ioc(OrderRequest("LON_FLY", bid, Side.SELL, sz))

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------
    def _send_ioc(self, order):
        resp = self.send_order(order)
        if resp and resp.filled < resp.volume:
            # Cancel unfilled remainder (true IOC behavior)
            self.cancel_order(resp.id)
        return resp

    def start_periodic_requote(self, interval=30):
        def loop():
            while True:
                time.sleep(interval)
                with self._lock:
                    snap = dict(self.theos)
                for p, t in snap.items():
                    if self.quote_limiter.can_act(f"quote_{p}"):
                        self.quote_limiter.mark(f"quote_{p}")
                        self._place_quotes(p, t)
        Thread(target=loop, daemon=True).start()
        print(f"[Bot] Periodic requote every {interval}s started.")

    def start_pnl_logger(self, interval=60):
        """Periodic PnL and position monitoring."""
        def loop():
            while True:
                time.sleep(interval)
                try:
                    pnl_data = self.get_pnl()
                    positions = self.get_positions()
                    active_pos = {k: v for k, v in positions.items() if v != 0}
                    if active_pos:
                        print(f"[Monitor] Positions: {active_pos}")
                    if pnl_data:
                        total = sum(v for v in pnl_data.values() if isinstance(v, (int, float)))
                        print(f"[Monitor] PnL total: {total:.0f}")
                except Exception as e:
                    print(f"[Monitor] Error: {e}")
        Thread(target=loop, daemon=True).start()


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = Config()

    bot = ImprovedTradingBot(cfg)

    print("=" * 60)
    print("  IMCity Trading Bot v4 — Max PnL Edition")
    print("=" * 60)
    print(f"  Exchange: {cfg.EXCHANGE_URL}")
    print(f"  User:     {cfg.USERNAME}")
    print(f"  Window:   {cfg.START_TIME} -> {cfg.SETTLEMENT_TIME}")
    print(f"  Max order size: {cfg.MAX_ORDER_SIZE}  |  Max position: ±{cfg.MAX_POSITION}")
    print(f"  Group A (×4): {cfg.GROUP_A}")
    print(f"  Theo-market blending: {cfg.BLEND_THEO_WITH_MARKET}")
    print("=" * 60)

    # Pre-authenticate BEFORE starting any threads
    # (cached_property is not thread-safe — avoids 403 race condition)
    print("[Bot] Authenticating...", end=" ")
    try:
        _ = bot.auth_token
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        print("[Bot] Cannot authenticate. Check username/password and exchange URL.")
        return

    # Start SSE stream first (uses auth_token), then background tasks
    bot.start()
    bot.start_background_tasks()
    bot.start_periodic_requote(interval=25)
    bot.start_pnl_logger(interval=60)

    print("[Bot] Running. Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Bot] Shutting down...")
        bot.cancel_all_orders()
        bot.stop()
        print("[Bot] Stopped.")


if __name__ == "__main__":
    main()
