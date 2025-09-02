# pip install yfinance pandas numpy xlsxwriter plotly
import numpy as np
import pandas as pd
import yfinance as yf
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

TICKER   = "BTC-USD"
OUT_XLSX = "btc_hourly.xlsx"
OUT_HTML = "btc_hourly_chart.html"
LOCAL_TZ = ZoneInfo("America/Phoenix")

# Indicator settings
EMA_SPAN       = 250   # 250-hour EMA
CCI_PERIOD     = 20    # CCI lookback
CCI_AVG_PERIOD = 20    # SMA of CCI

def fetch_btc_hourly():
    df = yf.download(
        TICKER,
        period="60d",      # Yahoo hourly history limit
        interval="1h",
        auto_adjust=True,
        prepost=False,
        progress=False,
        threads=True,
    )
    if df.empty:
        raise RuntimeError("No data returned for BTC-USD")

    # Flatten columns if MultiIndex (e.g. ('Close','BTC-USD') -> 'Close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert to local time and drop tz (Excel requirement)
    df.index = df.index.tz_convert(LOCAL_TZ).tz_localize(None)

    # ---- Indicators ----
    df["EMA_250"] = df["Close"].ewm(span=EMA_SPAN, adjust=False).mean()

    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_mean = tp.rolling(CCI_PERIOD, min_periods=CCI_PERIOD).mean()

    def _mad(x):
        return np.mean(np.abs(x - x.mean()))
    tp_mad = tp.rolling(CCI_PERIOD, min_periods=CCI_PERIOD).apply(_mad, raw=False)

    df["CCI"] = (tp - tp_mean) / (0.015 * tp_mad)
    df["CCI_MA"] = df["CCI"].rolling(CCI_AVG_PERIOD, min_periods=CCI_AVG_PERIOD).mean()

    # ---- Trend & Crosses ----
    df["Uptrend"] = df["EMA_250"] > df["EMA_250"].shift(1)
    cross_up   = (df["CCI"] > df["CCI_MA"]) & (df["CCI"].shift(1) <= df["CCI_MA"].shift(1))
    cross_down = (df["CCI"] < df["CCI_MA"]) & (df["CCI"].shift(1) >= df["CCI_MA"].shift(1))
    df["BuySignal"]  = (cross_up)   & df["Uptrend"]
    df["SellSignal"] = (cross_down) & df["Uptrend"]

    # Reset index for downstream work; keep a clean Datetime column
    df = df.reset_index().rename(columns={"index": "Datetime"})
    return df

def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """Create a trade log based on Buy/Sell signals (long-only, 1 unit)."""
    trades = []
    in_pos = False
    entry_i = None

    for i, row in df.iterrows():
        if not in_pos and bool(row["BuySignal"]):
            in_pos = True
            entry_i = i
        elif in_pos and bool(row["SellSignal"]):
            entry_time  = df.at[entry_i, "Datetime"]
            entry_price = float(df.at[entry_i, "Close"])
            exit_time   = row["Datetime"]
            exit_price  = float(row["Close"])
            profit      = exit_price - entry_price  # 1-unit PnL

            trades.append(
                {
                    "EntryIndex": entry_i,
                    "EntryTime": entry_time,
                    "EntryPrice": entry_price,
                    "ExitIndex": i,
                    "ExitTime": exit_time,
                    "ExitPrice": exit_price,
                    "TradeProfit": profit,
                }
            )
            in_pos = False
            entry_i = None

    return pd.DataFrame(trades)

def attach_pnl_to_bars(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Add TradeProfit (on exit bar), DailyProfit (by exit day), and TotalProfit (cumulative)."""
    df = df.copy()

    # Initialize columns
    df["TradeProfit"] = np.nan

    if trades.empty:
        # No trades; set daily/total to 0
        df["DailyProfit"] = 0.0
        df["TotalProfit"] = 0.0
        return df

    # Map trade PnL to its exit bar
    for _, tr in trades.iterrows():
        exit_idx = int(tr["ExitIndex"])
        df.at[exit_idx, "TradeProfit"] = float(tr["TradeProfit"])

    # Daily profit by exit date
    trades["ExitDate"] = pd.to_datetime(trades["ExitTime"]).dt.date
    daily = trades.groupby("ExitDate")["TradeProfit"].sum()

    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df["DailyProfit"] = df["Date"].map(daily).fillna(0.0)

    # Total profit (cumulative over time)
    df["TotalProfit"] = df["TradeProfit"].fillna(0.0).cumsum()

    return df

def make_interactive_chart(df: pd.DataFrame):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=("BTC-USD (1h) • EMA-250 + Signals",
                        "CCI (with SMA)",
                        "Equity Curve (Total Profit)")
    )

    # Price panel
    fig.add_trace(
        go.Candlestick(
            x=df["Datetime"], open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="BTC-USD"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Datetime"], y=df["EMA_250"],
            mode="lines", name="EMA 250", line=dict(width=1.3)
        ),
        row=1, col=1
    )

    # Buy/Sell markers (offset so they don't hide under candles)
    buy_y  = np.where(df["BuySignal"].astype(bool),  df["Low"]  * 0.995, np.nan)
    sell_y = np.where(df["SellSignal"].astype(bool), df["High"] * 1.005, np.nan)

    fig.add_trace(
        go.Scatter(
            x=df["Datetime"], y=buy_y, mode="markers", name="BUY",
            marker=dict(symbol="triangle-up", size=12),
            hovertemplate="BUY<br>%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df["Datetime"], y=sell_y, mode="markers", name="SELL",
            marker=dict(symbol="triangle-down", size=12),
            hovertemplate="SELL<br>%{x}<br>%{y:.2f}<extra></extra>",
        ),
        row=1, col=1
    )

    # CCI panel
    fig.add_trace(
        go.Scatter(x=df["Datetime"], y=df["CCI"], mode="lines",
                   name=f"CCI {CCI_PERIOD}", line=dict(width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df["Datetime"], y=df["CCI_MA"], mode="lines",
                   name=f"CCI SMA {CCI_AVG_PERIOD}", line=dict(width=1.2, dash="dash")),
        row=2, col=1
    )
    for yval, nm, dash in [(100, "+100", "dot"), (-100, "-100", "dot"), (0, "0", "dash")]:
        fig.add_hline(y=yval, line_width=1, line_dash=dash, line_color="gray",
                      annotation_text=nm, annotation_position="top left", row=2, col=1)

    # Equity curve panel
    fig.add_trace(
        go.Scatter(
            x=df["Datetime"], y=df["TotalProfit"],
            mode="lines", name="Total Profit", line=dict(width=2)
        ),
        row=3, col=1
    )
    # Optional: show daily profit as lollipop markers on the equity panel
    dp_mask = df["DailyProfit"] != 0
    fig.add_trace(
        go.Scatter(
            x=df.loc[dp_mask, "Datetime"],
            y=df.loc[dp_mask, "TotalProfit"],
            mode="markers", name="Daily Profit (exits)",
            marker=dict(size=6),
            hovertemplate="Exit Day PnL: %{customdata:.2f}<extra></extra>",
            customdata=df.loc[dp_mask, "DailyProfit"],
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        title="BTC-USD Hourly • Signals + P&L",
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    for r in [1, 2, 3]:
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", row=r, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="CCI", row=2, col=1)
    fig.update_yaxes(title_text="PnL", row=3, col=1)
    return fig

def main():
    df = fetch_btc_hourly()

    # Build trades + attach PnL columns to the bar data
    trade_log = build_trade_log(df)
    df = attach_pnl_to_bars(df, trade_log)

    print(f"Trades: {len(trade_log)} | Total PnL: {trade_log['TradeProfit'].sum() if not trade_log.empty else 0:.2f}")

    # Save Excel: main bars + signals + PnL, and a separate "Trades" sheet
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter",
                        datetime_format="yyyy-mm-dd hh:mm") as writer:
        df.to_excel(writer, sheet_name="BTC-USD", index=False)
        if not trade_log.empty:
            trade_log[[
                "EntryTime","EntryPrice","ExitTime","ExitPrice","TradeProfit"
            ]].to_excel(writer, sheet_name="Trades", index=False)

    # Save interactive chart
    fig = make_interactive_chart(df)
    pio.write_html(fig, file=OUT_HTML, auto_open=False, full_html=True, include_plotlyjs="inline")

    print(f"Saved: {OUT_XLSX}")
    print(f"Saved: {OUT_HTML}  (open this in your browser)")

if __name__ == "__main__":
    main()