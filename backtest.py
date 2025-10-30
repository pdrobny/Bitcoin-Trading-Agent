# btc_2y_coinbase_compounding_with_hodl.py
# pip install ccxt pandas numpy plotly xlsxwriter

import time
import numpy as np
import pandas as pd
import ccxt
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ---------- Config ----------
YEARS       = 2
TIMEFRAME   = "1h"
SYMBOL      = "BTC/USD"
EXCHANGE_ID = "coinbase"           # Coinbase only
LOCAL_TZ    = ZoneInfo("America/Phoenix")

OUT_XLSX    = "btc_backtest.xlsx"
OUT_HTML    = "btc_backtest.html"

# Indicators
EMA_SPAN       = 250 # default 250 
CCI_PERIOD     = 20 # default 20
CCI_AVG_PERIOD = 20 # default 20

# --- Long-only regime filter ---
LONG_ONLY_FILTER = True          # if True, only take BUYs when a higher-timeframe uptrend is confirmed
LONG_ONLY_EMA_SPAN = 500         # choose 250 or 500 to define the long-term trend baseline
LONG_ONLY_REQUIRE_PRICE_ABOVE = True   # require Close > chosen EMA
LONG_ONLY_REQUIRE_EMA_RISING = True    # require chosen EMA is rising


# --- EMA crossover params ---
FAST_EMA_SPAN = 250 # 250 and 500 best so far
SLOW_EMA_SPAN = 500

# Compounding position sizing
INITIAL_CAPITAL_USD = 10000.0   # starting equity
POSITION_FRACTION   = 1.0      # invest this fraction of current equity on each entry (1.0 = 100%)
FEE_BPS_PER_SIDE    = 0.0      # per-side fee in bps (e.g., 10 = 0.10%). Keep 0.0 to ignore.

# Buy-and-hold benchmark config
# Choose when we "buy and hold":
#   "first_entry" -> when the strategy first enters a trade
#   "first_bar"   -> at the first bar of the dataset
BUYHOLD_FROM = "first_entry"

# --- Uptrend definition tweak (optional) ---
REQUIRE_PRICE_ABOVE_EMA = True  # use Close > EMA_250 in addition to EMA rising

# --- Stop-loss for uptrend trades ---
STOP_LOSS_PCT       = 0.25     # 25% stop
USE_TRAILING_STOP   = True      # trail stop to lock gains during uptrend trades
STOP_ON_INTRABAR    = True      # fill at stop if Low pierces it intrabar

# ---------- Helpers ----------
def timeframe_ms(tf: str) -> int:
    units = {"m":60, "h":3600, "d":86400}
    n, u = int(tf[:-1]), tf[-1]
    return n * units[u] * 1000

def chunked_ranges(start_ms: int, end_ms: int, tf_ms: int, limit: int):
    step = tf_ms * limit
    out, s = [], start_ms
    while s < end_ms:
        e = min(s + step, end_ms)
        out.append((s, e))
        s = e
    return out

def fetch_all_ohlcv_coinbase(symbol: str, timeframe: str,
                             start_ms: int, end_ms: int,
                             limit_per_call: int = 300,
                             pause: float = 0.45) -> pd.DataFrame:
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    tf_ms = timeframe_ms(timeframe)
    windows = chunked_ranges(start_ms, end_ms, tf_ms, limit_per_call)
    rows = []

    for win_start, win_end in windows:
        since = win_start
        while since < win_end:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            nxt = last_ts + tf_ms
            if nxt <= since:
                break
            since = min(nxt, win_end)
            time.sleep(pause)

    if not rows:
        raise RuntimeError("No OHLCV returned from Coinbase.")
    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"]).drop_duplicates("ts")
    df["Datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    df = df.drop(columns=["ts"]).sort_values("Datetime").reset_index(drop=True)
    return df[["Datetime","Open","High","Low","Close","Volume"]]

def get_window_ms(years: int):
    end_utc = pd.Timestamp.utcnow()
    start_utc = end_utc - pd.DateOffset(years=years)
    return int(start_utc.timestamp()*1000), int(end_utc.timestamp()*1000)

# ---------- Indicators & Signals ----------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA_250"] = df["Close"].ewm(span=EMA_SPAN, adjust=False).mean()
    df["EMA_500"] = df["Close"].ewm(span=500, adjust=False).mean()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_mean = tp.rolling(CCI_PERIOD, min_periods=CCI_PERIOD).mean()
    def _mad(x): return np.mean(np.abs(x - x.mean()))
    tp_mad = tp.rolling(CCI_PERIOD, min_periods=CCI_PERIOD).apply(_mad, raw=False)
    df["CCI"] = (tp - tp_mean) / (0.015 * tp_mad)
    df["CCI_MA"] = df["CCI"].rolling(CCI_AVG_PERIOD, min_periods=CCI_AVG_PERIOD).mean()
    return df

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Regime: bull when EMA(250) > EMA(500)
    df["Bull"] = df["EMA_250"] > df["EMA_500"]

    # Regime cross events
    df["RegimeUp"]   = (df["EMA_250"] > df["EMA_500"]) & (df["EMA_250"].shift(1) <= df["EMA_500"].shift(1))   # golden cross
    df["RegimeDown"] = (df["EMA_250"] < df["EMA_500"]) & (df["EMA_250"].shift(1) >= df["EMA_500"].shift(1))   # death cross

    # CCI cross vs its MA (use your existing periods)
    cci_up   = (df["CCI"] > df["CCI_MA"]) & (df["CCI"].shift(1) <= df["CCI_MA"].shift(1))
    cci_down = (df["CCI"] < df["CCI_MA"]) & (df["CCI"].shift(1) >= df["CCI_MA"].shift(1))

    # Hybrid signals:
    # - In Bull regime: trade EMA crossovers (buy only on RegimeUp, sell on RegimeDown)
    # - Outside Bull:  trade CCI crossovers
    ema_buy  = df["RegimeUp"]
    ema_sell = df["RegimeDown"]
    cci_buy  = (~df["Bull"]) & cci_up
    cci_sell = (~df["Bull"]) & cci_down

    df["BuySignal"]  = ema_buy | cci_buy
    df["SellSignal"] = ema_sell | cci_sell

    # Keep detail columns for the engine/plot (optional but useful)
    df["CCIBuy"]  = cci_buy
    df["CCISell"] = cci_sell
    return df

# ---------- Trades & PnL (Compounding) ----------
def _apply_fees(entry_price: float, exit_price: float) -> tuple[float, float]:
    if FEE_BPS_PER_SIDE <= 0:
        return entry_price, exit_price
    fee = FEE_BPS_PER_SIDE / 10_000.0
    adj_entry = entry_price * (1 + fee)
    adj_exit  = exit_price * (1 - fee)
    return adj_entry, adj_exit

def build_trade_log_compounding(df: pd.DataFrame) -> pd.DataFrame:
    trades, in_pos, entry_i = [], False, None
    equity = INITIAL_CAPITAL_USD
    entry_mode = None  # "EMA" or "CCI"

    for i, row in df.iterrows():
        px_close = float(row["Close"])

        # -------- Entry --------
        if (not in_pos) and bool(row.get("BuySignal", False)):
            in_pos, entry_i = True, i
            entry_price = px_close
            adj_entry, _ = _apply_fees(entry_price, entry_price)

            alloc_usd = equity * POSITION_FRACTION
            qty = 0.0 if adj_entry <= 0 else alloc_usd / adj_entry
            df.at[entry_i, "SizedQty"] = qty
            df.at[entry_i, "EquityBefore"] = equity

            # mode determined by regime at entry
            entry_mode = "EMA" if bool(row.get("Bull", False)) else "CCI"
            continue

        # -------- Manage --------
        if in_pos:
            entry_price = float(df.at[entry_i, "Close"])
            qty = float(df.at[entry_i, "SizedQty"])

            exit_now, exit_price, reason = False, None, None

            if entry_mode == "EMA":
                # Exit EMA trades only on EMA death cross (RegimeDown)
                if bool(row.get("RegimeDown", False)):
                    exit_now, exit_price, reason = True, px_close, "EMA_XDOWN"

            else:  # entry_mode == "CCI"
                # Primary exit: CCI cross-down while outside Bull
                if bool(row.get("CCISell", False)):
                    exit_now, exit_price, reason = True, px_close, "CCI_XDOWN"
                # Also force exit on Bull regime flip (RegimeUp) so we can pivot into EMA trade next bar
                elif bool(row.get("RegimeUp", False)):
                    exit_now, exit_price, reason = True, px_close, "REGIME_UP_EXIT"

            if exit_now:
                adj_entry, adj_exit = _apply_fees(entry_price, exit_price)
                entry_val = qty * adj_entry
                exit_val  = qty * adj_exit
                pnl_usd   = exit_val - entry_val
                ret_pct   = (adj_exit / adj_entry - 1.0) * 100.0

                equity_before = float(df.at[entry_i, "EquityBefore"])
                equity_after  = equity_before + pnl_usd
                equity = equity_after

                trades.append({
                    "EntryIndex": entry_i,
                    "EntryTime":  df.at[entry_i, "Datetime"],
                    "EntryPrice": entry_price,
                    "ExitIndex":  i,
                    "ExitTime":   row["Datetime"],
                    "ExitPrice":  exit_price,
                    "Qty":        qty,
                    "EntryValueUSD": entry_val,
                    "ExitValueUSD":  exit_val,
                    "TradeProfitUSD": pnl_usd,
                    "ReturnPct":      ret_pct,
                    "EquityBefore":   equity_before,
                    "EquityAfter":    equity_after,
                    "ExitReason":     reason,
                    "EntryMode":      entry_mode,
                })

                in_pos, entry_i, entry_mode = False, None, None

    return pd.DataFrame(trades)

def attach_pnl_and_equity(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TradeProfitUSD"] = np.nan
    df["EquityUSD"]      = np.nan
    df["ExitReason"]     = ""      # <— NEW
    df["StopExit"]       = False   # <— NEW
    df["CCIExit"]        = False   # <— NEW

    # init equity
    df.loc[df.index[0], "EquityUSD"] = INITIAL_CAPITAL_USD

    if trades.empty:
        df["DailyProfitUSD"] = 0.0
        df["TotalProfitUSD"] = 0.0
        df["EquityUSD"] = df["EquityUSD"].ffill().fillna(INITIAL_CAPITAL_USD)
        df["BuyHoldEquityUSD"] = _build_buyhold(df, start_mode="first_bar")
        return df

    # map realized PnL, equity, and exit reason on exit bars
    for _, tr in trades.iterrows():
        exit_idx = int(tr["ExitIndex"])
        df.at[exit_idx, "TradeProfitUSD"] = float(tr["TradeProfitUSD"])
        df.at[exit_idx, "EquityUSD"]      = float(tr["EquityAfter"])
        reason = str(tr.get("ExitReason",""))
        df.at[exit_idx, "ExitReason"] = reason
        if reason == "STOP": df.at[exit_idx, "StopExit"] = True
        if reason == "CCI":  df.at[exit_idx, "CCIExit"]  = True

    # daily & cumulative PnL
    trades["ExitDate"] = pd.to_datetime(trades["ExitTime"]).dt.date
    daily = trades.groupby("ExitDate")["TradeProfitUSD"].sum()

    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df["DailyProfitUSD"] = df["Date"].map(daily).fillna(0.0)
    df["TotalProfitUSD"] = df["TradeProfitUSD"].fillna(0.0).cumsum()

    # equity curve (forward-fill)
    df["EquityUSD"] = df["EquityUSD"].ffill().fillna(INITIAL_CAPITAL_USD)

    # Buy & Hold benchmark
    start_mode = BUYHOLD_FROM
    if start_mode == "first_entry" and not trades.empty:
        start_idx = int(trades.iloc[0]["EntryIndex"])
    else:
        start_idx = 0
    df["BuyHoldEquityUSD"] = _build_buyhold(df, start_mode=None, start_idx=start_idx)

    return df

def _build_buyhold(df: pd.DataFrame, start_mode: str | None = None, start_idx: int | None = None) -> pd.Series:
    """
    Build a buy-and-hold equity series:
    - Entry at start_idx (or 0 if None), investing INITIAL_CAPITAL_USD once.
    - Apply entry fee only. Equity = qty * Close (mark-to-market).
    """
    if start_idx is None:
        start_idx = 0
    prices = df["Close"].astype(float).values
    if start_idx >= len(prices):
        start_idx = 0
    entry_price = prices[start_idx]
    # apply entry fee (conservative)
    if FEE_BPS_PER_SIDE > 0:
        entry_price = entry_price * (1 + FEE_BPS_PER_SIDE / 10_000.0)
    qty = 0.0 if entry_price <= 0 else INITIAL_CAPITAL_USD / entry_price

    equity = np.full(len(prices), np.nan, dtype=float)
    equity[:start_idx] = INITIAL_CAPITAL_USD  # before buying, just flat at initial capital
    equity[start_idx:] = qty * prices[start_idx:]
    return pd.Series(equity, index=df.index, name="BuyHoldEquityUSD")

# ---------- Chart ----------
def make_chart(df: pd.DataFrame):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.55,0.25,0.20],
                        subplot_titles=("BTC/USD (Coinbase, 1h) • EMA-250 + Signals",
                                        f"CCI ({CCI_PERIOD}) + SMA({CCI_AVG_PERIOD})",
                                        "Equity Curve (USD): Strategy vs Buy&Hold"))
    
    # Price + EMA
    fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="BTC/USD"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_250"], mode="lines", name="EMA 250"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_500"], mode="lines", name="EMA 500"), row=1, col=1)

    # Signal markers (offset for visibility)
    buy_y  = np.where(df["BuySignal"].astype(bool), df["Low"]*0.995,  np.nan)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=buy_y, mode="markers", name="BUY",
                             marker=dict(symbol="triangle-up", size=10)), row=1,col=1)

    stop_y = np.where(df["StopExit"].astype(bool), df["High"]*1.006, np.nan)
    sell_y  = np.where(df["SellSignal"].astype(bool),  df["High"]*1.012, np.nan)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=stop_y, mode="markers", name="STOP EXIT",
                             marker=dict(symbol="x", size=10)), row=1,col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=sell_y, mode="markers", name="SELL",
                             marker=dict(symbol="triangle-down", size=10)), row=1,col=1)


    # CCI panel
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["CCI"], mode="lines", name="CCI"), row=2,col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["CCI_MA"], mode="lines", name="CCI MA",
                             line=dict(dash="dash")), row=2,col=1)
    for yval, nm, dash in [(100, "+100", "dot"), (-100, "-100", "dot"), (0, "0", "dash")]:
        fig.add_hline(y=yval, line_width=1, line_dash=dash, line_color="gray",
                      annotation_text=nm, annotation_position="top left", row=2, col=1)

    # Equity curves (USD)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EquityUSD"], mode="lines",
                             name="Strategy Equity (USD)"), row=3,col=1)
    if "BuyHoldEquityUSD" in df.columns:
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BuyHoldEquityUSD"], mode="lines",
                                 name="Buy & Hold Equity (USD)"), row=3,col=1)

    fig.update_layout(title="BTC/USD Hourly • 2 Years (Compounding vs Buy & Hold)",
                      xaxis_rangeslider_visible=True, hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="CCI",   row=2, col=1)
    fig.update_yaxes(title_text="Equity (USD)", row=3, col=1)
    return fig

# ---------- Main ----------
def main():
    start_ms, end_ms = get_window_ms(YEARS)
    print(f"Fetching {YEARS}y of {SYMBOL} {TIMEFRAME} from {EXCHANGE_ID}…")
    df = fetch_all_ohlcv_coinbase(SYMBOL, TIMEFRAME, start_ms, end_ms)
    print(f"Rows: {len(df)} | Range: {df['Datetime'].min()} → {df['Datetime'].max()}")

    bars = add_indicators(df)
    bars = add_signals(bars)
    trades = build_trade_log_compounding(bars)
    bars = attach_pnl_and_equity(bars, trades)

    # --- Exports ---
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as w:
        bars.to_excel(w, sheet_name="BTCUSD", index=False)
        if not trades.empty:
            trades[["EntryTime","EntryPrice","ExitTime","ExitPrice","Qty",
                    "EntryValueUSD","ExitValueUSD","TradeProfitUSD","ReturnPct",
                    "EquityBefore","EquityAfter"]].to_excel(
                w, sheet_name="Trades", index=False
            )

    fig = make_chart(bars)
    pio.write_html(fig, file=OUT_HTML, auto_open=False, full_html=True, include_plotlyjs="inline")
    print(f"Saved: {OUT_XLSX}")
    print(f"Saved: {OUT_HTML}")

    # Quick summary
    strat_final = float(bars["EquityUSD"].ffill().iloc[-1])
    bh_final    = float(bars["BuyHoldEquityUSD"].ffill().iloc[-1]) if "BuyHoldEquityUSD" in bars else INITIAL_CAPITAL_USD
    print(f"Final Strategy Equity: ${strat_final:,.2f} | Final Buy&Hold Equity: ${bh_final:,.2f}")

if __name__ == "__main__":
    main()