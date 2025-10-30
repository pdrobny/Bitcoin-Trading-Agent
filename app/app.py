# ---- .env loader (robust on Windows/IDEs) ----
import os
from pathlib import Path

def _load_env():
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        raise RuntimeError("python-dotenv not installed. Run: pip install python-dotenv")

    # Try common locations in order:
    candidates = [
        Path(__file__).with_name(".env"),   # same folder as app.py
        Path.cwd() / ".env",                # current working dir
        Path(find_dotenv(usecwd=True) or ""),  # whatever python-dotenv finds
    ]
    loaded_from = None
    for p in candidates:
        if p and str(p) != "" and Path(p).exists():
            load_dotenv(dotenv_path=str(p), override=False)
            loaded_from = str(p)
            break

    # Optional: print where we loaded from and current CWD (safe; doesn’t leak secrets)
    print(f"[env] CWD={Path.cwd()}")
    print(f"[env] Loaded .env from: {loaded_from if loaded_from else 'NONE'}")

_load_env()

def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}. Ensure it is set or present in your .env")
    return v
ALPACA_API_KEY     = require_env("ALPACA_API_KEY")
ALPACA_API_SECRET  = require_env("ALPACA_API_SECRET")


# app.py
#import os
import threading
import time
import math
import queue
import logging
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
#from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, render_template, send_file

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.common.exceptions import APIError
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

# ----------------------------
# Config from ENV
# ----------------------------
def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}. "
                           f"Ensure it’s set or present in your .env")
    return v


ALPACA_PAPER       = os.getenv("ALPACA_PAPER", "true").lower() == "true"
SYMBOL               = os.getenv("SYMBOL", "BTC/USD")
LOCAL_TZ             = ZoneInfo(os.getenv("LOCAL_TZ", "America/Phoenix"))
FEE_BUFFER_PCT       = float(os.getenv("FEE_BUFFER_PCT", "0.001"))  # 0.1%
SLEEP_SEC            = int(os.getenv("SLEEP_SEC", "15"))
LOOKBACK_BARS        = int(os.getenv("LOOKBACK_BARS", "600"))

# Optional Gemini (non-blocking)
USE_GEMINI           = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp-02-05")
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")

# Email (Gmail App Password required)
GMAIL_ADDRESS        = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD   = os.getenv("GMAIL_APP_PASSWORD", "")
EOD_TO_EMAILS        = [e.strip() for e in os.getenv("EOD_TO_EMAILS", "").split(",") if e.strip()]

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TRADES_CSV = DATA_DIR / "trades.csv"
EQUITY_CSV = DATA_DIR / "equity.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# Alpaca
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
data_client    = CryptoHistoricalDataClient()

# ----------------------------
# State
# ----------------------------
run_flag = {"running": False}
state = {
    "last_bar_time": None,
    "position_qty": 0.0,
    "position_side": "flat",
    "last_signal": None,
    "last_action": None,
    "last_error": None,
    "regime": None,
    "updated_at": None,
    "entry_price": None,      # track last entry price for realized P&L calc
    "entry_qty": 0.0,
    "realized_pnl": 0.0       # total realized P&L since app start
}

# ----------------------------
# Indicators
# ----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma = tp.rolling(length).mean()
    md  = (tp - sma).abs().rolling(length).mean()
    return (tp - sma) / (0.015 * md)

def cci_sma(cci_series: pd.Series, length: int = 20) -> pd.Series:
    return cci_series.rolling(length).mean()

def crossed_above(a_prev, a_curr, b_prev, b_curr) -> bool:
    return a_prev <= b_prev and a_curr > b_curr

def crossed_below(a_prev, a_curr, b_prev, b_curr) -> bool:
    return a_prev >= b_prev and a_curr < b_curr

# ----------------------------
# Data fetching
# ----------------------------
def fetch_hourly_bars(symbol: str, limit: int) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=70)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Hour, start=start, end=end, limit=limit)
    bars = data_client.get_crypto_bars(req).df
    if bars.empty:
        return pd.DataFrame()
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(symbol, level="symbol")
    bars = bars.tz_convert(LOCAL_TZ)
    return bars.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})

# ----------------------------
# Trading helpers
# ----------------------------
def get_equity_usd() -> float:
    a = trading_client.get_account()
    return float(a.equity)

def alpaca_symbol_no_slash(symbol: str) -> str:
    return symbol.replace("/", "")

def get_position_qty(symbol: str) -> float:
    """
    Robust across Alpaca crypto symbols. Prefers get_open_position(),
    falls back to scanning all positions. Returns 0.0 if flat.
    """
    sym_noslash = alpaca_symbol_no_slash(symbol)
    # 1) Preferred: get_open_position("BTCUSD")
    try:
        p = trading_client.get_open_position(sym_noslash)
        return float(p.qty)
    except Exception:
        pass

    # 2) Try with slash (some setups work this way)
    try:
        p = trading_client.get_open_position(symbol)
        return float(p.qty)
    except Exception:
        pass

    # 3) Fallback: scan all positions
    try:
        positions = trading_client.get_all_positions()
        for p in positions:
            if getattr(p, "symbol", "") in (symbol, sym_noslash):
                return float(p.qty)
    except Exception:
        pass

    return 0.0

def submit_sell_all(symbol: str):
    qty = get_position_qty(symbol)
    if qty <= 0:
        return None
    order = MarketOrderRequest(
        symbol=symbol,     # keep "BTC/USD" for orders
        qty=str(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    return trading_client.submit_order(order)


# ----------------------------
# Persistence (trades & equity)
# ----------------------------
def append_trade(ts: datetime, action: str, signal: str, price: float, qty: float,
                 equity_before: float, equity_after: float, pnl_realized: float, note: str = ""):
    row = {
        "timestamp": ts.isoformat(),
        "action": action,
        "signal": signal,
        "price": price,
        "qty": qty,
        "equity_before": equity_before,
        "equity_after": equity_after,
        "pnl_realized": pnl_realized,
        "note": note
    }
    df = pd.DataFrame([row])
    header = not TRADES_CSV.exists()
    df.to_csv(TRADES_CSV, mode="a", index=False, header=header)

def append_equity_snapshot(ts: datetime, equity: float, price: float, regime: str):
    row = {"timestamp": ts.isoformat(), "equity": equity, "price": price, "regime": regime}
    df = pd.DataFrame([row])
    header = not EQUITY_CSV.exists()
    df.to_csv(EQUITY_CSV, mode="a", index=False, header=header)

def load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        return pd.DataFrame(columns=["timestamp","action","signal","price","qty","equity_before","equity_after","pnl_realized","note"])
    return pd.read_csv(TRADES_CSV)

def load_equity() -> pd.DataFrame:
    if not EQUITY_CSV.exists():
        return pd.DataFrame(columns=["timestamp","equity","price","regime"])
    return pd.read_csv(EQUITY_CSV)

# ----------------------------
# Gemini (optional explanation)
# ----------------------------
def gemini_explain_decision(context: dict) -> None:
    if not USE_GEMINI or not GEMINI_API_KEY:
        return
    try:
        from google.generativeai import configure, GenerativeModel
        configure(api_key=GEMINI_API_KEY)
        prompt = f"""
You are a trading assistant. Given this context, produce a one-sentence rationale and a concise action tag.

Context JSON:
{context}

Rules:
- Strategy fixed: bull=EMA250>EMA500 acts on EMA cross; nonbull uses CCI(20) cross vs its SMA(20).
- Output:
rationale: <short>
action: <BUY|SELL|HOLD>
"""
        model = GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        log.info(f"[Gemini] {text}")
    except Exception as e:
        log.warning(f"Gemini explanation failed: {e}")

# ----------------------------
# Core signal logic (closed bars only)
# ----------------------------
def evaluate_and_trade():
    try:
        bars = fetch_hourly_bars(SYMBOL, LOOKBACK_BARS)
        if bars.empty or len(bars) < 510:
            state["last_error"] = "Insufficient bars for EMA(500)."
            return

        last_time = bars.index[-1]
        if state["last_bar_time"] == last_time:
            return  # already processed

        close = bars["close"]
        price = float(close.iloc[-1])

        ema250 = ema(close, 250)
        ema500 = ema(close, 500)

        dfhlc = bars[["high","low","close"]].copy()
        cci20 = cci(dfhlc, 20)
        cci20_sma = cci_sma(cci20, 20)

        e250_prev, e500_prev = float(ema250.iloc[-2]), float(ema500.iloc[-2])
        e250_curr, e500_curr = float(ema250.iloc[-1]), float(ema500.iloc[-1])
        cci_prev, cci_sma_prev = float(cci20.iloc[-2]), float(cci20_sma.iloc[-2])
        cci_curr, cci_sma_curr = float(cci20.iloc[-1]), float(cci20_sma.iloc[-1])

        bull = e250_curr > e500_curr
        regime = "bull" if bull else "nonbull"

        ema_cross_up   = crossed_above(e250_prev, e250_curr, e500_prev, e500_curr)
        ema_cross_down = crossed_below(e250_prev, e250_curr, e500_prev, e500_curr)
        cci_cross_up   = crossed_above(cci_prev, cci_curr, cci_sma_prev, cci_sma_curr)
        cci_cross_down = crossed_below(cci_prev, cci_curr, cci_sma_prev, cci_sma_curr)

        signal = "none"
        action = "hold"
        if regime == "bull":
            if ema_cross_up:
                signal, action = "ema_cross_up", "buy"
            elif ema_cross_down:
                signal, action = "ema_cross_down", "sell"
        else:
            if cci_cross_up:
                signal, action = "cci_cross_up", "buy"
            elif cci_cross_down:
                signal, action = "cci_cross_down", "sell"

        qty_now = get_position_qty(SYMBOL)
        equity_before = get_equity_usd()
        notional = max(0.0, equity_before * (1.0 - FEE_BUFFER_PCT))

        order_resp = None
        realized_pnl_this = 0.0
        trade_qty = 0.0

        if action == "buy" and qty_now <= 0:
            order_resp = submit_buy_notional(SYMBOL, notional)
            state["position_side"] = "long"
            # approximate qty using notional/price (Alpaca fills may differ slightly)
            trade_qty = round(notional / price, 8)
            state["entry_price"] = price
            state["entry_qty"] = trade_qty

        elif action == "sell" and qty_now > 0:
            order_resp = submit_sell_all(SYMBOL)
            state["position_side"] = "flat"
            trade_qty = state["entry_qty"] if state["entry_qty"] else qty_now
            if state["entry_price"] and trade_qty:
                realized_pnl_this = (price - state["entry_price"]) * trade_qty
                state["realized_pnl"] += realized_pnl_this
            state["entry_price"] = None
            state["entry_qty"] = 0.0

        # Update state & logs
        state["last_bar_time"] = last_time
        state["position_qty"]  = get_position_qty(SYMBOL)
        state["last_signal"]   = signal
        state["last_action"]   = action
        state["regime"]        = regime
        state["updated_at"]    = datetime.now(LOCAL_TZ).isoformat()
        state["last_error"]    = None

        # Equity after (best effort)
        equity_after = get_equity_usd()

        # Log trade row only when we actually sent an order
        if order_resp is not None:
            append_trade(
                ts=last_time, action=action, signal=signal, price=price, qty=trade_qty,
                equity_before=equity_before, equity_after=equity_after,
                pnl_realized=realized_pnl_this,
                note=f"order_id={getattr(order_resp, 'id', '')}"
            )

        # Snapshot equity once per processed bar (for chart)
        append_equity_snapshot(ts=last_time, equity=equity_after, price=price, regime=regime)

        # Optional Gemini explanation (non-blocking)
        if USE_GEMINI:
            ctx = dict(
                symbol=SYMBOL, last_bar=str(last_time), regime=regime,
                ema250_prev=e250_prev, ema250_curr=e250_curr,
                ema500_prev=e500_prev, ema500_curr=e500_curr,
                cci_prev=cci_prev, cci_curr=cci_curr,
                cci_sma_prev=cci_sma_prev, cci_sma_curr=cci_sma_curr,
                equity_before=equity_before, equity_after=equity_after,
                action=action, signal=signal
            )
            threading.Thread(target=gemini_explain_decision, args=(ctx,), daemon=True).start()

    except Exception as e:
        log.exception("evaluate_and_trade error")
        state["last_error"] = str(e)
        state["updated_at"] = datetime.now(LOCAL_TZ).isoformat()

# ----------------------------
# Worker
# ----------------------------
def worker_loop():
    log.info("Worker started.")
    try:
        state["position_qty"] = get_position_qty(SYMBOL)
    except Exception as e:
        state["last_error"] = str(e)
    while run_flag["running"]:
        evaluate_and_trade()
        time.sleep(SLEEP_SEC)
    log.info("Worker stopped.")

# ----------------------------
# Email helper
# ----------------------------
def send_email(subject: str, body: str, attachments: list[Path] = None):
    if not (GMAIL_ADDRESS and GMAIL_APP_PASSWORD and EOD_TO_EMAILS):
        raise RuntimeError("Missing GMAIL_ADDRESS/GMAIL_APP_PASSWORD/EOD_TO_EMAILS.")
    msg = EmailMessage()
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ", ".join(EOD_TO_EMAILS)
    msg["Subject"] = subject
    msg.set_content(body)

    for path in (attachments or []):
        with open(path, "rb") as f:
            data = f.read()
        maintype, subtype = ("application", "octet-stream")
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)

    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)

# ----------------------------
# Flask endpoints
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html", symbol=SYMBOL, local_tz=str(LOCAL_TZ))

@app.route("/health")
def health():
    return jsonify(ok=True, now=datetime.now(LOCAL_TZ).isoformat())

@app.route("/status")
def status():
    s = state.copy()
    s["running"] = run_flag["running"]
    return jsonify(s)

@app.route("/trades.json")
def trades_json():
    df = load_trades().tail(200)  # last 200 rows
    return df.to_json(orient="records")

@app.route("/equity.json")
def equity_json():
    df = load_equity().tail(1000)
    return df.to_json(orient="records")

@app.route("/download/trades.csv")
def download_trades_csv():
    if not TRADES_CSV.exists():
        return jsonify(error="No trades yet"), 404
    return send_file(TRADES_CSV, as_attachment=True, download_name="trades.csv")

@app.route("/download/trades.xlsx")
def download_trades_xlsx():
    df = load_trades()
    if df.empty:
        return jsonify(error="No trades yet"), 404
    path = DATA_DIR / "trades.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="Trades")
    return send_file(path, as_attachment=True, download_name="trades.xlsx")

@app.route("/start", methods=["POST","GET"])
def start():
    if run_flag["running"]:
        return jsonify(message="already running"), 200
    run_flag["running"] = True
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
    return jsonify(message="started"), 200

@app.route("/stop", methods=["POST","GET"])
def stop():
    if not run_flag["running"]:
        return jsonify(message="already stopped"), 200
    run_flag["running"] = False
    return jsonify(message="stopping"), 200

@app.route("/debug/send-eod", methods=["POST","GET"])
def debug_send_eod():
    """Email last week's trades with summary: daily, weekly, overall P&L."""
    try:
        now = datetime.now(LOCAL_TZ)
        df = load_trades()
        if df.empty:
            return jsonify(message="No trades to email"), 200

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)

        # Realized PnL per trade (already computed), aggregate
        df["pnl_realized"] = pd.to_numeric(df["pnl_realized"], errors="coerce").fillna(0.0)

        today = now.date()
        week_ago = today - timedelta(days=7)

        df_week = df[df["timestamp"].dt.date >= week_ago]

        daily_pnl = df[df["date"] == today]["pnl_realized"].sum()
        weekly_pnl = df_week["pnl_realized"].sum()
        overall_pnl = df["pnl_realized"].sum()

        # Export CSV/XLSX attachments (weekly window for brevity)
        csv_path = DATA_DIR / f"trades_last7d_{today}.csv"
        xlsx_path = DATA_DIR / f"trades_last7d_{today}.xlsx"
        df_week.to_csv(csv_path, index=False)
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
            df_week.to_excel(xw, index=False, sheet_name="Last7dTrades")

        subject = f"[BTC Bot] EOD {today} — Daily: {daily_pnl:+.2f} | Weekly: {weekly_pnl:+.2f} | Overall: {overall_pnl:+.2f}"
        body = (
            f"Symbol: {SYMBOL}\n"
            f"Date: {today}\n"
            f"Daily Realized P&L: {daily_pnl:+.2f}\n"
            f"Weekly Realized P&L (last 7d): {weekly_pnl:+.2f}\n"
            f"Overall Realized P&L: {overall_pnl:+.2f}\n"
            f"Running status: {'RUNNING' if run_flag['running'] else 'STOPPED'}\n"
            f"Regime: {state.get('regime')}\n"
            f"Last signal/action: {state.get('last_signal')}/{state.get('last_action')}\n"
        )
        send_email(subject, body, attachments=[csv_path, xlsx_path])
        return jsonify(message="EOD email sent", subject=subject), 200
    except Exception as e:
        log.exception("/debug/send-eod failed")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
