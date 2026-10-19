import time
import math
import requests
import pandas as pd
import numpy as np
from threading import Thread
from datetime import datetime
from scipy.optimize import curve_fit
from bot_template import BaseBot, OrderRequest, Side

# ==========================================
# 1. THE ALPHA ENGINE
# ==========================================
class AlphaEngine:
    def __init__(self, aerodatabox_key=""):
        self.aerodatabox_key = aerodatabox_key
        self.london_lat = 51.5074
        self.london_lon = -0.1278
        
        # Define the exact strict settlement window
        self.start_time = pd.to_datetime("2026-02-28 12:00:00").tz_localize("Europe/London")
        self.settlement_time = pd.to_datetime("2026-03-01 12:00:00").tz_localize("Europe/London")

    def get_weather_data(self):
        """Fetches blended past realized actuals and future forecasts."""
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.london_lat,
            "longitude": self.london_lon,
            "minutely_15": "temperature_2m,relative_humidity_2m,wind_speed_10m,visibility",
            "past_minutely_15": 48,
            "forecast_minutely_15": 96,
            "timezone": "Europe/London"
        }
        try:
            resp = requests.get(url, params=params)
            if not resp.ok: return None
            
            data = resp.json()["minutely_15"]
            df = pd.DataFrame({
                "time": pd.to_datetime(data["time"]).dt.tz_localize("Europe/London"),
                "temp_c": data["temperature_2m"],
                "humidity": data["relative_humidity_2m"],
                "wind_kph": data["wind_speed_10m"],
                "visibility_m": data["visibility"]
            })
            
            df = df[(df["time"] >= self.start_time) & (df["time"] <= self.settlement_time)].copy()
            df["temp_f"] = df["temp_c"] * 9/5 + 32
            df["wx_value"] = df["temp_f"] * df["humidity"]
            return df
        except Exception as e:
            print(f"Weather Data Error: {e}")
            return None

    def calculate_weather_theos(self, weather_df):
        if weather_df is None or weather_df.empty: return None, None
        wx_sum_theo = (weather_df["wx_value"] / 100).sum()
        spot_row = weather_df[weather_df["time"] == self.settlement_time]
        wx_spot_theo = spot_row["wx_value"].values[0] if not spot_row.empty else 0
        return wx_spot_theo, wx_sum_theo

    def get_tidal_theos(self, weather_df):
        url = "https://environment.data.gov.uk/flood-monitoring/id/measures/0006-level-tidal_level-i-15_min-mAOD/readings"
        try:
            resp = requests.get(url, params={"_sorted": "", "_limit": 400})
            if not resp.ok: return None, None
            
            items = resp.json().get("items", [])
            df = pd.DataFrame(items)[["dateTime", "value"]]
            df["time"] = pd.to_datetime(df["dateTime"], utc=True).dt.tz_convert("Europe/London")
            df["level"] = df["value"].astype(float)
            df = df.sort_values("time").dropna()
            
            t_hours = (df["time"] - df["time"].min()).dt.total_seconds() / 3600.0
            y_data = df["level"].values
            
            def tidal_wave(t, a, omega, phi, c):
                return a * np.sin(omega * t + phi) + c
                
            p0 = [3.0, 2 * np.pi / 12.42, 0.0, 0.0]
            popt, _ = curve_fit(tidal_wave, t_hours, y_data, p0=p0)
            
            target_hours = (self.settlement_time - df["time"].min()).total_seconds() / 3600.0
            base_prediction = tidal_wave(target_hours, *popt)
            
            # Storm Surge Residual
            spot_weather = weather_df[weather_df["time"] == self.settlement_time]
            wind_kph = spot_weather["wind_kph"].values[0] if not spot_weather.empty else 0
            storm_surge = (wind_kph - 25.0) * 0.02 if wind_kph > 25.0 else 0.0
                
            tide_spot_theo = abs(base_prediction + storm_surge) * 1000

            # TIDE_SWING Realized + Forecasted
            realized_df = df[(df["time"] >= self.start_time)].copy()
            realized_df["diff_cm"] = realized_df["level"].diff().abs() * 100
            
            def strangle_payoff(diff):
                if pd.isna(diff): return 0
                return max(0, 20 - diff) + max(0, diff - 25)
                
            realized_strangle = realized_df["diff_cm"].apply(strangle_payoff).sum()
            
            now = pd.Timestamp.now(tz="Europe/London")
            future_times = pd.date_range(start=now, end=self.settlement_time, freq="15min")
            future_hours = (future_times - df["time"].min()).total_seconds() / 3600.0
            future_levels = tidal_wave(future_hours, *popt)
            future_diffs = np.abs(np.diff(future_levels)) * 100
            forecast_strangle = sum(strangle_payoff(d) for d in future_diffs)
            
            tide_swing_theo = (realized_strangle + forecast_strangle) * 100 
            return tide_spot_theo, tide_swing_theo
        except Exception as e:
            print(f"Tidal error: {e}")
            return None, None

    def get_flight_theos(self, weather_df):
        if not self.aerodatabox_key: return None, None
        
        now = pd.Timestamp.now(tz="Europe/London")
        elapsed_minutes = int((now - self.start_time).total_seconds() / 60)
        fetch_minutes = min(elapsed_minutes, 720) 
        
        url = "https://aerodatabox.p.rapidapi.com/flights/airports/iata/LHR"
        headers = {"x-rapidapi-host": "aerodatabox.p.rapidapi.com", "x-rapidapi-key": self.aerodatabox_key}
        params = {"offsetMinutes": -fetch_minutes, "durationMinutes": fetch_minutes, "direction": "Both"}
        
        try:
            resp = requests.get(url, headers=headers, params=params)
            if not resp.ok: return None, None
            
            data = resp.json()
            realized_arr = len(data.get("arrivals", []))
            realized_dep = len(data.get("departures", []))
            realized_total = realized_arr + realized_dep
            
            # Weather Dampening
            future_weather = weather_df[weather_df["time"] > now]
            poor_vis = len(future_weather[future_weather["visibility_m"] < 2000])
            high_wind = len(future_weather[future_weather["wind_kph"] > 40])
            
            run_rate_per_min = realized_total / max(elapsed_minutes, 1)
            remaining_minutes = int((self.settlement_time - now).total_seconds() / 60)
            dampener = max(0.4, 1.0 - (poor_vis * 0.05) - (high_wind * 0.05))
            
            forecast_total = run_rate_per_min * remaining_minutes * dampener
            lhr_count_theo = realized_total + forecast_total
            imbalance = 100 * (realized_arr - realized_dep) / max(realized_total, 1)
            
            return lhr_count_theo, abs(imbalance)
        except Exception as e:
            print(f"Flight error: {e}")
            return None, None


# ==========================================
# 2. THE UNIFIED TRADING BOT
# ==========================================
class UnifiedTradingBot(BaseBot):
    def __init__(self, cmi_url, username, password, api_key):
        super().__init__(cmi_url, username, password)
        self.alpha = AlphaEngine(aerodatabox_key=api_key)
        self.theos = {}
        self.latest_books = {}
        
        self.quote_width = 4.0
        self.order_size = 2
        self.max_position = 100

    def start_background_tasks(self):
        def fetch_loop():
            flight_counter = 0
            while True:
                weather_df = self.alpha.get_weather_data()
                if weather_df is not None:
                    wx_spot, wx_sum = self.alpha.calculate_weather_theos(weather_df)
                    if wx_spot: self.theos["WX_SPOT"] = wx_spot
                    if wx_sum: self.theos["WX_SUM"] = wx_sum
                    
                    tide_spot, tide_swing = self.alpha.get_tidal_theos(weather_df)
                    if tide_spot: self.theos["TIDE_SPOT"] = tide_spot
                    if tide_swing: self.theos["TIDE_SWING"] = tide_swing
                    
                    if flight_counter % 3 == 0:
                        lhr_count, lhr_index = self.alpha.get_flight_theos(weather_df)
                        if lhr_count: self.theos["LHR_COUNT"] = lhr_count
                        if lhr_index: self.theos["LHR_INDEX"] = lhr_index
                    
                    if all(k in self.theos for k in ["WX_SPOT", "TIDE_SPOT", "LHR_COUNT"]):
                        self.theos["LON_ETF"] = self.theos["WX_SPOT"] + self.theos["TIDE_SPOT"] + self.theos["LHR_COUNT"]
                        self.calculate_options_theo()
                    
                    print(f"[*] Theos Updated: {self.theos}")
                flight_counter += 1
                time.sleep(300) 

        t = Thread(target=fetch_loop, daemon=True)
        t.start()

    def calculate_options_theo(self):
        """Calculates exact payoff of the LON_FLY package based on our ETF prediction."""
        S = self.theos.get("LON_ETF", 0)
        if S == 0: return
        
        # Package payoff formula
        p_6200_put = max(0, 6200 - S)
        c_6200_call = max(0, S - 6200)
        c_6600_call = max(0, S - 6600)
        c_7000_call = max(0, S - 7000)
        
        self.theos["LON_FLY"] = (2 * p_6200_put) + c_6200_call - (2 * c_6600_call) + (3 * c_7000_call)

    def on_trades(self, trade):
        action = "BOUGHT" if trade.buyer == self.username else "SOLD"
        print(f"[FILL] {action} {trade.volume}x {trade.product} @ {trade.price}")

    def on_orderbook(self, ob):
        self.latest_books[ob.product] = ob
        
        if ob.product == "LON_FLY":
            self.run_options_arbitrage()
            
        if ob.product in ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]:
            self.run_etf_arbitrage()
            
        self.update_quotes(ob.product)

    def run_options_arbitrage(self):
        """Attacks mispriced LON_FLY options if they deviate from our theoretical payoff."""
        if "LON_FLY" not in self.latest_books or "LON_FLY" not in self.theos: return
        
        book = self.latest_books["LON_FLY"]
        theo = self.theos["LON_FLY"]
        margin = 5.0 # Required profit margin to execute
        
        if book.sell_orders and book.sell_orders[0].price < theo - margin:
            print(f"[ARB] Option UNDERVALUED. Buying LON_FLY at {book.sell_orders[0].price} (Theo: {theo})")
            self.send_ioc(OrderRequest("LON_FLY", book.sell_orders[0].price, Side.BUY, 1))
            
        elif book.buy_orders and book.buy_orders[0].price > theo + margin:
            print(f"[ARB] Option OVERVALUED. Selling LON_FLY at {book.buy_orders[0].price} (Theo: {theo})")
            self.send_ioc(OrderRequest("LON_FLY", book.buy_orders[0].price, Side.SELL, 1))

    def run_etf_arbitrage(self):
        req = ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
        if not all(p in self.latest_books for p in req): return

        etf_book = self.latest_books["LON_ETF"]
        tide_book = self.latest_books["TIDE_SPOT"]
        wx_book = self.latest_books["WX_SPOT"]
        lhr_book = self.latest_books["LHR_COUNT"]

        if not (etf_book.buy_orders and tide_book.sell_orders and wx_book.sell_orders and lhr_book.sell_orders): return

        etf_bid = etf_book.buy_orders[0].price
        sum_asks = tide_book.sell_orders[0].price + wx_book.sell_orders[0].price + lhr_book.sell_orders[0].price

        if etf_bid > sum_asks + 2:
            print(f"[ARB] Executing! ETF Bid ({etf_bid}) > Component Asks ({sum_asks})")
            self.send_ioc(OrderRequest("LON_ETF", etf_bid, Side.SELL, 1))
            self.send_ioc(OrderRequest("TIDE_SPOT", tide_book.sell_orders[0].price, Side.BUY, 1))
            self.send_ioc(OrderRequest("WX_SPOT", wx_book.sell_orders[0].price, Side.BUY, 1))
            self.send_ioc(OrderRequest("LHR_COUNT", lhr_book.sell_orders[0].price, Side.BUY, 1))

    def send_ioc(self, order):
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            self.cancel_order(resp.id)

    def update_quotes(self, product):
        if product not in self.theos: return
            
        theo = self.theos[product]
        current_pos = self.get_positions().get(product, 0)
        
        skew = (current_pos / self.max_position) * 3.0 
        skewed_theo = theo - skew
        
        bid_price = math.floor(skewed_theo - self.quote_width)
        ask_price = math.ceil(skewed_theo + self.quote_width)

        self.cancel_all_orders_for_product(product)
        if current_pos < self.max_position:
            self.send_order(OrderRequest(product, bid_price, Side.BUY, self.order_size))
        if current_pos > -self.max_position:
            self.send_order(OrderRequest(product, ask_price, Side.SELL, self.order_size))

    def cancel_all_orders_for_product(self, product):
        orders = self.get_orders(product)
        for o in orders:
            self.cancel_order(o["id"])


if __name__ == "__main__":
    # Add your credentials here
    EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
    USERNAME = "particle accelerator"
    PASSWORD = "lebronisgoat"
    RAPID_API_KEY = "6c94f344c9msh1c1f381bea99cbdp1e310fjsn9bd2fcb93d70" # Copied from your screenshot

    bot = UnifiedTradingBot(EXCHANGE_URL, USERNAME, PASSWORD, RAPID_API_KEY)
    
    print("Initializing Alpha Engine and connecting to Exchange...")
    bot.start_background_tasks()
    bot.start() 
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down... cancelling active orders.")
        bot.cancel_all_orders()
        bot.stop()