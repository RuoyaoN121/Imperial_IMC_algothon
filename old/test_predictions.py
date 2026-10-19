"""
IMCity Bot — Test Script
=========================
Run this BEFORE connecting to the exchange to verify your predictions.

Stage 1: Just fetch data and print theos (no exchange needed)
Stage 2: Compare theos to live market prices (needs exchange)

Usage:
    python test_predictions.py              # Stage 1 only (no exchange)
    python test_predictions.py --exchange   # Stage 1 + Stage 2
"""

import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Import the bot components
from improved_bot_v3 import Config, AlphaEngine, TradeTracker, EdgeCalculator
from bot_template import BaseBot, OrderBook, Side

# ============================================================
# CONFIG — fill in your details
# ============================================================
cfg = Config()
cfg.EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
cfg.USERNAME = "YOUR_USERNAME"       # TODO: fill in
cfg.PASSWORD = "YOUR_PASSWORD"       # TODO: fill in
cfg.RAPID_API_KEY = "YOUR_API_KEY"   # TODO: fill in (or leave as-is to skip flights)


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_section(text):
    print(f"\n--- {text} ---")


# ============================================================
# STAGE 1: Check predictions (no exchange needed)
# ============================================================
def stage_1():
    print_header("STAGE 1: Checking Data Sources & Predictions")

    alpha = AlphaEngine(cfg)

    # --- Weather ---
    print_section("Weather Data")
    wx_df = alpha.fetch_weather()
    if wx_df is not None:
        print(f"  ✓ Got {len(wx_df)} weather readings")
        print(f"  Time range: {wx_df['time'].min()} → {wx_df['time'].max()}")

        # Show current conditions
        now = pd.Timestamp.now(tz="Europe/London")
        current = wx_df.iloc[(wx_df["time"] - now).abs().argsort().iloc[0]]
        print(f"\n  Current conditions (nearest to now):")
        print(f"    Time:        {current['time']}")
        print(f"    Temp:        {current['temp_c']:.1f}°C / {current['temp_c']*9/5+32:.1f}°F")
        print(f"    Humidity:    {current['humidity']:.0f}%")
        print(f"    Wind:        {current['wind_kph']:.1f} kph")
        print(f"    Visibility:  {current['visibility_m']:.0f} m")

        # Show settlement forecast
        settle_rows = wx_df[wx_df["time"] == alpha.settlement_time]
        if not settle_rows.empty:
            s = settle_rows.iloc[0]
            print(f"\n  Settlement time forecast ({alpha.settlement_time}):")
            print(f"    Temp:        {s['temp_c']:.1f}°C / {s['temp_c']*9/5+32:.1f}°F")
            print(f"    Humidity:    {s['humidity']:.0f}%")
            print(f"    Wind:        {s['wind_kph']:.1f} kph")
        else:
            print(f"\n  ⚠ No forecast row for exact settlement time")

        # Calculate theos
        wx_spot = alpha.calc_wx_spot()
        wx_sum = alpha.calc_wx_sum()
        print(f"\n  → WX_SPOT theo:  {wx_spot:.0f}" if wx_spot else "\n  ✗ WX_SPOT: failed")
        print(f"  → WX_SUM theo:   {wx_sum:.0f}" if wx_sum else "  ✗ WX_SUM: failed")
    else:
        print("  ✗ Failed to fetch weather data")

    # --- Tides ---
    print_section("Tidal Data")
    tide_df = alpha.fetch_tides()
    if tide_df is not None:
        print(f"  ✓ Got {len(tide_df)} tidal readings")
        print(f"  Time range: {tide_df['time'].min()} → {tide_df['time'].max()}")
        print(f"  Current level: {tide_df['level'].iloc[-1]:.3f} mAOD")
        print(f"  Recent range: {tide_df['level'].min():.3f} to {tide_df['level'].max():.3f} mAOD")

        if alpha._tide_model_params is not None:
            a, omega, phi, c = alpha._tide_model_params
            period_hours = 2 * np.pi / omega
            print(f"\n  Tidal model fit:")
            print(f"    Amplitude:  {a:.3f} m")
            print(f"    Period:     {period_hours:.2f} hours")
            print(f"    Offset:     {c:.3f} m")

        tide_spot = alpha.calc_tide_spot()
        tide_swing = alpha.calc_tide_swing()
        print(f"\n  → TIDE_SPOT theo:  {tide_spot:.0f}" if tide_spot else "\n  ✗ TIDE_SPOT: failed")
        print(f"  → TIDE_SWING theo: {tide_swing:.0f}" if tide_swing else "  ✗ TIDE_SWING: failed")
    else:
        print("  ✗ Failed to fetch tidal data")

    # --- Flights ---
    print_section("Flight Data")
    if cfg.RAPID_API_KEY and cfg.RAPID_API_KEY != "YOUR_API_KEY":
        lhr_count, lhr_index = alpha.fetch_and_calc_flights()
        if lhr_count is not None:
            print(f"  → LHR_COUNT theo: {lhr_count:.0f}")
            print(f"  → LHR_INDEX theo: {lhr_index:.1f}")
        else:
            print("  ✗ Flight fetch failed (check API key)")
    else:
        print("  ⚠ Skipped — no AeroDataBox API key set")
        lhr_count, lhr_index = None, None

    # --- Derived ---
    print_section("Derived Products")
    tide_spot = alpha.calc_tide_spot()
    wx_spot = alpha.calc_wx_spot()

    if all(v is not None for v in [tide_spot, wx_spot, lhr_count]):
        etf = alpha.calc_etf(tide_spot, wx_spot, lhr_count)
        fly = alpha.calc_options(etf)
        print(f"  → LON_ETF theo:  {etf:.0f}")
        print(f"  → LON_FLY theo:  {fly:.0f}")

        print(f"\n  ETF breakdown:")
        print(f"    TIDE_SPOT: {tide_spot:.0f}")
        print(f"    WX_SPOT:   {wx_spot:.0f}")
        print(f"    LHR_COUNT: {lhr_count:.0f}")
        print(f"    Total:     {tide_spot + wx_spot + lhr_count:.0f}")
    else:
        print("  ⚠ Cannot compute ETF/FLY — missing component theos")
        missing = []
        if tide_spot is None: missing.append("TIDE_SPOT")
        if wx_spot is None: missing.append("WX_SPOT")
        if lhr_count is None: missing.append("LHR_COUNT")
        print(f"    Missing: {', '.join(missing)}")

    # --- Summary ---
    print_section("FULL THEO SUMMARY")
    theos = alpha.compute_all_theos()
    if lhr_count is not None: theos["LHR_COUNT"] = lhr_count
    if lhr_index is not None: theos["LHR_INDEX"] = lhr_index

    tide_spot = theos.get("TIDE_SPOT")
    wx_spot = theos.get("WX_SPOT")
    lhr = theos.get("LHR_COUNT")
    if all(v is not None for v in [tide_spot, wx_spot, lhr]):
        theos["LON_ETF"] = tide_spot + wx_spot + lhr
        theos["LON_FLY"] = alpha.calc_options(theos["LON_ETF"])

    print(f"  {'Product':<15} {'Theo':>10}")
    print(f"  {'-'*25}")
    for product in ["TIDE_SPOT", "TIDE_SWING", "WX_SPOT", "WX_SUM",
                     "LHR_COUNT", "LHR_INDEX", "LON_ETF", "LON_FLY"]:
        val = theos.get(product)
        print(f"  {product:<15} {val:>10.0f}" if val is not None else f"  {product:<15} {'N/A':>10}")

    # Sanity checks
    print_section("SANITY CHECKS")
    issues = []

    if tide_spot is not None and (tide_spot < 0 or tide_spot > 5000):
        issues.append(f"TIDE_SPOT={tide_spot:.0f} seems out of range (expected 0-5000)")
    if wx_spot is not None and (wx_spot < 500 or wx_spot > 8000):
        issues.append(f"WX_SPOT={wx_spot:.0f} seems out of range (expected 500-8000)")
    if lhr_count is not None and (lhr_count < 100 or lhr_count > 2000):
        issues.append(f"LHR_COUNT={lhr_count:.0f} seems out of range (expected 100-2000)")
    if theos.get("LON_ETF") and (theos["LON_ETF"] < 2000 or theos["LON_ETF"] > 15000):
        issues.append(f"LON_ETF={theos['LON_ETF']:.0f} seems out of range")

    if issues:
        for i in issues:
            print(f"  ⚠ {i}")
    else:
        print("  ✓ All theos look reasonable")

    return theos


# ============================================================
# STAGE 2: Compare to live market (needs exchange connection)
# ============================================================
def stage_2(theos):
    print_header("STAGE 2: Comparing Theos to Live Market")

    class ReadOnlyBot(BaseBot):
        def on_orderbook(self, ob): pass
        def on_trades(self, trade): pass

    try:
        bot = ReadOnlyBot(cfg.EXCHANGE_URL, cfg.USERNAME, cfg.PASSWORD)
        products = bot.get_products()
        print(f"  ✓ Connected to exchange. {len(products)} products found.\n")
    except Exception as e:
        print(f"  ✗ Failed to connect: {e}")
        print(f"    Check your username/password and that the exchange is running.")
        return

    # Get orderbooks and recent trades
    print(f"  {'Product':<13} {'Theo':>8} {'Mid':>8} {'Spread':>8} {'Gap':>8} {'Gap%':>8} {'Status'}")
    print(f"  {'-'*70}")

    for p in products:
        sym = p.symbol
        theo = theos.get(sym)

        try:
            ob = bot.get_orderbook(sym)
        except Exception:
            print(f"  {sym:<13} {'?':>8} {'error':>8}")
            continue

        if ob.buy_orders and ob.sell_orders:
            best_bid = ob.buy_orders[0].price
            best_ask = ob.sell_orders[0].price
            mid = (best_bid + best_ask) / 2
            spread = best_ask - best_bid

            if theo is not None:
                gap = theo - mid
                gap_pct = gap / mid * 100 if mid != 0 else 0

                if abs(gap_pct) < 1:
                    status = "✓ aligned"
                elif abs(gap_pct) < 5:
                    status = "~ moderate gap"
                else:
                    status = "⚠ BIG GAP"

                print(f"  {sym:<13} {theo:>8.0f} {mid:>8.0f} {spread:>8.0f} "
                      f"{gap:>+8.0f} {gap_pct:>+7.1f}% {status}")
            else:
                print(f"  {sym:<13} {'N/A':>8} {mid:>8.0f} {spread:>8.0f}")
        else:
            theo_str = f"{theo:.0f}" if theo is not None else "N/A"
            print(f"  {sym:<13} {theo_str:>8} {'no book':>8}")

    # Also show the ETF arb check
    print_section("ETF Arbitrage Check")
    try:
        etf_ob = bot.get_orderbook("LON_ETF")
        tide_ob = bot.get_orderbook("TIDE_SPOT")
        wx_ob = bot.get_orderbook("WX_SPOT")
        lhr_ob = bot.get_orderbook("LHR_COUNT")

        print(f"  Direction 1: Sell ETF, Buy Components")
        if etf_ob.buy_orders and tide_ob.sell_orders and wx_ob.sell_orders and lhr_ob.sell_orders:
            etf_bid = etf_ob.buy_orders[0].price
            comp_ask = tide_ob.sell_orders[0].price + wx_ob.sell_orders[0].price + lhr_ob.sell_orders[0].price
            edge1 = etf_bid - comp_ask
            print(f"    ETF bid: {etf_bid}  |  Component asks: {comp_ask}  |  Edge: {edge1:+.0f}")
            if edge1 > 3:
                print(f"    → ✓ ARB OPPORTUNITY! Profit per unit: {edge1:.0f}")
            else:
                print(f"    → No arb (need edge > 3)")
        else:
            print(f"    → Missing orderbook data")

        print(f"\n  Direction 2: Buy ETF, Sell Components")
        if etf_ob.sell_orders and tide_ob.buy_orders and wx_ob.buy_orders and lhr_ob.buy_orders:
            etf_ask = etf_ob.sell_orders[0].price
            comp_bid = tide_ob.buy_orders[0].price + wx_ob.buy_orders[0].price + lhr_ob.buy_orders[0].price
            edge2 = comp_bid - etf_ask
            print(f"    ETF ask: {etf_ask}  |  Component bids: {comp_bid}  |  Edge: {edge2:+.0f}")
            if edge2 > 3:
                print(f"    → ✓ ARB OPPORTUNITY! Profit per unit: {edge2:.0f}")
            else:
                print(f"    → No arb (need edge > 3)")
        else:
            print(f"    → Missing orderbook data")
    except Exception as e:
        print(f"  Error checking arb: {e}")

    # Show recent trades
    print_section("Recent Market Activity (last trades)")
    try:
        trades = bot.get_market_trades()
        if trades:
            # Group by product, show last price
            latest = {}
            for t in trades:
                latest[t.product] = t

            print(f"  {'Product':<13} {'Last Price':>10} {'Volume':>8}")
            print(f"  {'-'*35}")
            for sym in sorted(latest.keys()):
                t = latest[sym]
                print(f"  {sym:<13} {t.price:>10} {t.volume:>8}")
        else:
            print("  No recent trades found")
    except Exception as e:
        print(f"  Could not fetch trades: {e}")

    print_section("VERDICT")
    print("  If theos are within ~5% of market mids: your model is reasonable.")
    print("  If gap > 10%: investigate that product's model before trading.")
    print("  If books are empty: market making is very viable (wide spreads).")
    print("  If arb edges exist: the ETF arb strategy will print money.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    use_exchange = "--exchange" in sys.argv

    theos = stage_1()

    if use_exchange:
        stage_2(theos)
    else:
        print(f"\n{'='*60}")
        print(f"  To also compare against live market prices, run:")
        print(f"  python test_predictions.py --exchange")
        print(f"{'='*60}")

    print("\nDone.")
