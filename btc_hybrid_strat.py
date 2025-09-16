# hybrid_btc_simple.py
# Hybrid regime strategy:
#   • Bull (EMA250 > EMA500): trade EMA250/EMA500 crossovers
#   • Non-bull: trade CCI crossovers (CCI vs its SMA)
#
# Exports:
#   • btc_hybrid.xlsx (Bars + Trades)
#   • btc_hybrid_simple.html (simple price chart)
#   • btc_hybrid_full.html   (3-panel chart: Price, CCI, Equity)
#
# Run: python hybrid_btc_simple.py

import time
import numpy as np
import pandas as pd
import ccxt
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ==============================
# Config
# ==============================
YEARS       = 2
TIMEFRAME   = "1h"
SYMBOL      = "BTC/USD"
EXCHANGE_ID = "coinbase"
LOCAL_TZ    = ZoneInfo("America/Phoenix")

# Indicators
EMA_FAST_LONG  = 250
EMA_SLOW_LONG  = 500
CCI_PERIOD     = 20
CCI_AVG_PERIOD = 20

# Backtest / Sizing
INITIAL_CAPITAL_USD = 10_000.0
POSITION_FRACTION   = 1.0      # 1.0 = all-in; try 0.5, etc.
FEE_BPS_PER_SIDE    = 0.0      # e.g., 10 = 0.10% per side

# Output
OUT_HTML_SIMPLE = "btc_hybrid_simple.html"
OUT_HTML_FULL   = "btc_hybrid_full.html"
OUT_XLSX        = "btc_hybrid.xlsx"

# --- Execution behavior ---
ALLOW_SAME_BAR_PIVOT = True   # same-bar exit→entry when regime flips



# ==============================
# Data utils
# ==============================
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

def get_window_ms(years: int):
    end_utc = pd.Timestamp.utcnow()
    start_utc = end_utc - pd.DateOffset(years=years)
    return int(start_utc.timestamp()*1000), int(end_utc.timestamp()*1000)

def fetch_ohlcv_coinbase(symbol: str, timeframe: str,
                         start_ms: int, end_ms: int,
                         limit_per_call: int = 300, pause: float = 0.45) -> pd.DataFrame:
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
    tf_ms = timeframe_ms(timeframe)
    windows = chunked_ranges(start_ms, end_ms, tf_ms, limit_per_call)
    rows = []

    for w0, w1 in windows:
        since = w0
        while since < w1:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1][0]
            nxt = last_ts + tf_ms
            if nxt <= since:
                break
            since = min(nxt, w1)
            time.sleep(pause)

    if not rows:
        raise RuntimeError("No OHLCV returned.")
    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"]).drop_duplicates("ts")
    df["Datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    df = df.drop(columns=["ts"]).sort_values("Datetime").reset_index(drop=True)
    return df[["Datetime","Open","High","Low","Close","Volume"]]


# ==============================
# Indicators & Signals
# ==============================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Long EMAs for regime
    df["EMA_250"] = df["Close"].ewm(span=EMA_FAST_LONG, adjust=False).mean()
    df["EMA_500"] = df["Close"].ewm(span=EMA_SLOW_LONG, adjust=False).mean()

    # CCI and its SMA
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_mean = tp.rolling(CCI_PERIOD, min_periods=CCI_PERIOD).mean()

    # Mean absolute deviation
    def _mad(x):
        return np.mean(np.abs(x - x.mean()))
    tp_mad = tp.rolling(CCI_PERIOD, min_periods=CCI_PERIOD).apply(_mad, raw=False)
    denom = 0.015 * tp_mad.replace(0, np.nan)
    df["CCI"] = (tp - tp_mean) / denom
    df["CCI_MA"] = df["CCI"].rolling(CCI_AVG_PERIOD, min_periods=CCI_AVG_PERIOD).mean()

    return df

def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Regime
    df["Bull"] = df["EMA_250"] > df["EMA_500"]

    # EMA regime cross events
    df["RegimeUp"]   = (df["EMA_250"] > df["EMA_500"]) & (df["EMA_250"].shift(1) <= df["EMA_500"].shift(1))
    df["RegimeDown"] = (df["EMA_250"] < df["EMA_500"]) & (df["EMA_250"].shift(1) >= df["EMA_500"].shift(1))

    # --- NEW: if the dataset starts in bull regime, fire an initial RegimeUp signal ---
    if len(df) > 0 and df["Bull"].iloc[0]:
        df.at[df.index[0], "RegimeUp"] = True

    # CCI cross vs its SMA (non-bull trades)
    cci_up   = (df["CCI"] > df["CCI_MA"]) & (df["CCI"].shift(1) <= df["CCI_MA"].shift(1))
    cci_down = (df["CCI"] < df["CCI_MA"]) & (df["CCI"].shift(1) >= df["CCI_MA"].shift(1))

    # Hybrid signals
    ema_buy  = df["RegimeUp"]
    ema_sell = df["RegimeDown"]
    cci_buy  = (~df["Bull"]) & cci_up
    cci_sell = (~df["Bull"]) & cci_down

    df["BuySignal"]  = ema_buy | cci_buy
    df["SellSignal"] = ema_sell | cci_sell

    # For reference
    df["CCIBuy"]  = cci_buy
    df["CCISell"] = cci_sell
    return df

# ==============================
# Backtest (compounding, 1 position)
# ==============================
def _apply_fees(entry_price: float, exit_price: float) -> tuple[float, float]:
    if FEE_BPS_PER_SIDE <= 0:
        return entry_price, exit_price
    fee = FEE_BPS_PER_SIDE / 10_000.0
    return entry_price * (1 + fee), exit_price * (1 - fee)

def build_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades, in_pos, entry_i = [], False, None
    equity = INITIAL_CAPITAL_USD
    entry_mode = None  # "EMA" or "CCI"

    for i, row in df.iterrows():
        px_close = float(row["Close"])

        # Signals on this bar
        bull        = bool(row.get("Bull", False))
        regime_up   = bool(row.get("RegimeUp", False))
        regime_down = bool(row.get("RegimeDown", False))
        cci_buy     = bool(row.get("CCIBuy", False))
        cci_sell    = bool(row.get("CCISell", False))

        # ---------- Manage open position ----------
        if in_pos:
            should_exit, exit_price, reason = False, None, None

            if entry_mode == "EMA":
                if regime_down:
                    should_exit, exit_price, reason = True, px_close, "EMA_XDOWN"

            else:  # entry_mode == "CCI"
                if regime_up:
                    should_exit, exit_price, reason = True, px_close, "REGIME_UP_EXIT"
                elif cci_sell:
                    should_exit, exit_price, reason = True, px_close, "CCI_XDOWN"

            if should_exit:
                # close trade
                entry_price = float(df.at[entry_i, "Close"])
                qty = float(df.at[entry_i, "SizedQty"])
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

                # reset position state
                in_pos, entry_i, entry_mode = False, None, None

                # ---------- Same-bar pivot (optional) ----------
                if ALLOW_SAME_BAR_PIVOT:
                    # CCI → EMA on RegimeUp (immediate re-entry)
                    if reason == "REGIME_UP_EXIT" and regime_up:
                        # open EMA trade now
                        adj_entry, _ = _apply_fees(px_close, px_close)
                        qty = 0.0 if adj_entry <= 0 else (equity * POSITION_FRACTION) / adj_entry
                        in_pos, entry_i, entry_mode = True, i, "EMA"
                        df.at[entry_i, "SizedQty"] = qty
                        df.at[entry_i, "EquityBefore"] = equity
                        continue  # proceed to next bar

                    # EMA → CCI on RegimeDown (only if CCI buy also fires)
                    if reason == "EMA_XDOWN" and (not bull) and cci_buy:
                        adj_entry, _ = _apply_fees(px_close, px_close)
                        qty = 0.0 if adj_entry <= 0 else (equity * POSITION_FRACTION) / adj_entry
                        in_pos, entry_i, entry_mode = True, i, "CCI"
                        df.at[entry_i, "SizedQty"] = qty
                        df.at[entry_i, "EquityBefore"] = equity
                        continue  # proceed to next bar

                # if no pivot, just continue scanning
                continue

            # still in position, no exit this bar
            continue

        # ---------- Flat: handle entries ----------
        if bool(row.get("BuySignal", False)):
            adj_entry, _ = _apply_fees(px_close, px_close)
            qty = 0.0 if adj_entry <= 0 else (equity * POSITION_FRACTION) / adj_entry
            in_pos, entry_i = True, i
            entry_mode = "EMA" if bull else "CCI"
            df.at[entry_i, "SizedQty"] = qty
            df.at[entry_i, "EquityBefore"] = equity
            continue

    return pd.DataFrame(trades)

def attach_equity(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TradeProfitUSD"] = np.nan
    df["EquityUSD"]      = np.nan
    df.loc[df.index[0], "EquityUSD"] = INITIAL_CAPITAL_USD

    if trades.empty:
        df["EquityUSD"] = df["EquityUSD"].ffill().fillna(INITIAL_CAPITAL_USD)
        df["BuyHoldEquityUSD"] = _buyhold(df)  # buy at first bar
        return df

    for _, tr in trades.iterrows():
        exit_idx = int(tr["ExitIndex"])
        df.at[exit_idx, "TradeProfitUSD"] = float(tr["TradeProfitUSD"])
        df.at[exit_idx, "EquityUSD"]      = float(tr["EquityAfter"])

    df["EquityUSD"] = df["EquityUSD"].ffill().fillna(INITIAL_CAPITAL_USD)
    df["BuyHoldEquityUSD"] = _buyhold(df)
    return df

def _buyhold(df: pd.DataFrame) -> pd.Series:
    prices = df["Close"].astype(float).values
    entry_price = prices[0]
    if FEE_BPS_PER_SIDE > 0:
        entry_price = entry_price * (1 + FEE_BPS_PER_SIDE / 10_000.0)
    qty = INITIAL_CAPITAL_USD / entry_price
    return pd.Series(qty * prices, index=df.index, name="BuyHoldEquityUSD")


# ==============================
# Charts
# ==============================
def plot_chart_simple(df: pd.DataFrame):
    """Single-panel price chart with EMAs and buy/sell markers."""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    # Price
    fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="BTC/USD"), row=1, col=1)
    # EMAs
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_250"], mode="lines", name="EMA 250"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_500"], mode="lines", name="EMA 500"), row=1, col=1)
    # Markers
    buy_y  = np.where(df["BuySignal"].astype(bool),  df["Low"]*0.995,  np.nan)
    sell_y = np.where(df["SellSignal"].astype(bool), df["High"]*1.005, np.nan)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=buy_y,  mode="markers", name="BUY",
                             marker=dict(symbol="triangle-up", size=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=sell_y, mode="markers", name="SELL",
                             marker=dict(symbol="triangle-down", size=9)), row=1, col=1)
    fig.update_layout(title="BTC/USD • Hybrid (Price Only)",
                      xaxis_rangeslider_visible=True, hovermode="x unified")
    fig.update_yaxes(title_text="Price")
    return fig

def plot_chart_full(df: pd.DataFrame):
    """Three-panel chart: Price+EMAs, CCI panel, Equity curves."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            "BTC/USD (Coinbase, 1h) • EMA-250/500 + Signals",
            f"CCI ({CCI_PERIOD}) + SMA({CCI_AVG_PERIOD})",
            "Equity Curve (USD): Strategy vs Buy & Hold"
        )
    )

    # Price + EMAs
    fig.add_trace(go.Candlestick(x=df["Datetime"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="BTC/USD"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_250"], mode="lines", name="EMA 250"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EMA_500"], mode="lines", name="EMA 500"), row=1, col=1)

    # Buy/Sell markers
    buy_y  = np.where(df["BuySignal"].astype(bool),  df["Low"]*0.995,  np.nan)
    sell_y = np.where(df["SellSignal"].astype(bool), df["High"]*1.005, np.nan)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=buy_y,  mode="markers", name="BUY",
                             marker=dict(symbol="triangle-up", size=9)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=sell_y, mode="markers", name="SELL",
                             marker=dict(symbol="triangle-down", size=9)), row=1, col=1)

    # CCI panel
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["CCI"], mode="lines", name="CCI"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["CCI_MA"], mode="lines",
                             name="CCI MA", line=dict(dash="dash")), row=2, col=1)
    for yval, nm, dash in [(100, "+100", "dot"), (-100, "-100", "dot"), (0, "0", "dash")]:
        fig.add_hline(y=yval, line_width=1, line_dash=dash, line_color="gray",
                      annotation_text=nm, annotation_position="top left", row=2, col=1)

    # Equity panel
    fig.add_trace(go.Scatter(x=df["Datetime"], y=df["EquityUSD"], mode="lines",
                             name="Strategy Equity (USD)"), row=3, col=1)
    if "BuyHoldEquityUSD" in df.columns:
        fig.add_trace(go.Scatter(x=df["Datetime"], y=df["BuyHoldEquityUSD"], mode="lines",
                                 name="Buy & Hold Equity (USD)"), row=3, col=1)

    fig.update_layout(title="BTC/USD • Hybrid: EMA(250/500) in Bull, CCI in Non-Bull",
                      xaxis_rangeslider_visible=True, hovermode="x unified")
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="CCI",   row=2, col=1)
    fig.update_yaxes(title_text="Equity (USD)", row=3, col=1)
    return fig


# ==============================
# Main
# ==============================
def main():
    start_ms, end_ms = get_window_ms(YEARS)
    print(f"Fetching {YEARS}y of {SYMBOL} {TIMEFRAME} from {EXCHANGE_ID}…")
    df = fetch_ohlcv_coinbase(SYMBOL, TIMEFRAME, start_ms, end_ms)
    print(f"Rows: {len(df)} | Range: {df['Datetime'].min()} → {df['Datetime'].max()}")

    bars = add_indicators(df)
    bars = add_signals(bars)
    trades = build_trades(bars)
    bars = attach_equity(bars, trades)

    # Summary
    strat_final = float(bars["EquityUSD"].ffill().iloc[-1])
    bh_final    = float(bars["BuyHoldEquityUSD"].ffill().iloc[-1])
    print(f"Trades: {len(trades)}")
    print(f"Final Strategy Equity: ${strat_final:,.2f}")
    print(f"Final Buy&Hold Equity: ${bh_final:,.2f}")

    # --- Excel export ---
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as w:
        bars_cols = [
            "Datetime","Open","High","Low","Close","Volume",
            "EMA_250","EMA_500","CCI","CCI_MA",
            "Bull","RegimeUp","RegimeDown",
            "BuySignal","SellSignal",
            "EquityUSD","BuyHoldEquityUSD","TradeProfitUSD"
        ]
        bars.loc[:, bars_cols].to_excel(w, sheet_name="Bars", index=False)

        if not trades.empty:
            trade_cols = [
                "EntryTime","EntryPrice","ExitTime","ExitPrice","Qty",
                "EntryValueUSD","ExitValueUSD","TradeProfitUSD","ReturnPct",
                "EquityBefore","EquityAfter","EntryMode","ExitReason"
            ]
            trades.loc[:, trade_cols].to_excel(w, sheet_name="Trades", index=False)
    print(f"Saved Excel: {OUT_XLSX}")

    # --- Charts ---
    fig_full = plot_chart_full(bars)
    pio.write_html(fig_full, file=OUT_HTML_FULL, auto_open=False,
                   full_html=True, include_plotlyjs="inline")
    print(f"Saved chart: {OUT_HTML_FULL}")

if __name__ == "__main__":
    main()
