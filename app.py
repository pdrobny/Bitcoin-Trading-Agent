from pathlib import Path

# Flask/Jinja
from flask import Flask, jsonify, request, render_template, send_file, make_response
from jinja2 import ChoiceLoader, FileSystemLoader

# Std libs
import os
import json
import threading
import time
import logging
import smtplib
import math  # >>> NEW
from email.message import EmailMessage
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Third-party
import pandas as pd

# ---------- .env loader ----------
def _load_env():
    try:
        from dotenv import load_dotenv, find_dotenv
    except ImportError:
        raise RuntimeError("python-dotenv not installed. Run: pip install python-dotenv")
    p = find_dotenv(usecwd=True)
    if p:
        load_dotenv(p, override=False)
    here_dotenv = Path(__file__).with_name(".env")
    if here_dotenv.exists():
        load_dotenv(here_dotenv, override=False)
_load_env()

def require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

ALPACA_API_KEY     = require_env("ALPACA_API_KEY")
ALPACA_API_SECRET  = require_env("ALPACA_API_SECRET")

# ---------- config ----------
ALPACA_PAPER       = os.getenv("ALPACA_PAPER", "true").lower() == "true"
SYMBOL             = os.getenv("SYMBOL", "BTC/USD")
LOCAL_TZ           = ZoneInfo(os.getenv("LOCAL_TZ", "America/Phoenix"))
FEE_BUFFER_PCT     = float(os.getenv("FEE_BUFFER_PCT", "0.001"))  # 0.1%
SLEEP_SEC          = int(os.getenv("SLEEP_SEC", "15"))
LOOKBACK_BARS      = int(os.getenv("LOOKBACK_BARS", "600"))

# Failsafe: 25%
DRAW_DOWN_LIMIT    = float(os.getenv("DRAW_DOWN_LIMIT", "0.25"))

USE_GEMINI         = os.getenv("USE_GEMINI", "false").lower() == "true"
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp-02-05")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY", "")

TRADE_DECIDER      = os.getenv("TRADE_DECIDER", "rules").strip().lower()

# Buy sizing guards
BUY_BP_PCT         = float(os.getenv("BUY_BP_PCT", "0.98"))      # use at most 98% of buying power
MIN_NOTIONAL_USD   = float(os.getenv("MIN_NOTIONAL_USD", "5.0")) # avoid dust

# >>> NEW (tunable safety for notional)
NOTIONAL_SAFETY_PCT = float(os.getenv("NOTIONAL_SAFETY_PCT", "0.001"))  # 0.10% extra headroom
NOTIONAL_SAFETY_USD = float(os.getenv("NOTIONAL_SAFETY_USD", "25"))     # $25 absolute buffer
RETRY_BACKOFF_PCT   = float(os.getenv("RETRY_BACKOFF_PCT", "0.005"))    # 0.5% per retry
RETRY_MAX_ATTEMPTS  = int(os.getenv("RETRY_MAX_ATTEMPTS", "3"))

# Email / EOD
GMAIL_ADDRESS      = os.getenv("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
EOD_TO_EMAILS      = [e.strip() for e in os.getenv("EOD_TO_EMAILS", "").split(",") if e.strip()]
DAILY_EOD_HOUR     = int(os.getenv("DAILY_EOD_HOUR", "17"))
DAILY_EOD_MINUTE   = int(os.getenv("DAILY_EOD_MINUTE", "0"))

# ---------- paths ----------
DATA_DIR      = Path(os.getenv("DATA_DIR", "data"))
TRADES_CSV    = DATA_DIR / "trades.csv"
EQUITY_CSV    = DATA_DIR / "equity.csv"
ERRORS_CSV    = DATA_DIR / "errors.csv"
OPEN_TRADE    = DATA_DIR / "open_trade.json"
BASELINE_JSON = DATA_DIR / "baseline.json"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TRADELOG_COLS = [
    "EntryTime","EntryPrice","ExitTime","ExitPrice","Qty",
    "EntryValue","ExitValue","PnL_Realized","EntrySignal","ExitSignal","Note"
]

# ---------- Flask app ----------
TEMPLATES_ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(TEMPLATES_ROOT))
extra_template_paths = [
    TEMPLATES_ROOT,
    TEMPLATES_ROOT / "templates",
    Path.cwd(),
    Path.cwd() / "templates",
]
app.jinja_loader = ChoiceLoader([FileSystemLoader(str(p)) for p in extra_template_paths])

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# ---------- Alpaca clients ----------
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=ALPACA_PAPER)
data_client    = CryptoHistoricalDataClient()

# ---------- state ----------
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

    "entry_time": None,
    "entry_price": None,
    "entry_qty": 0.0,

    "realized_pnl": 0.0,
    "indicators": {},
    "llm": {
        "enabled": bool(USE_GEMINI and GEMINI_API_KEY),
        "model": GEMINI_MODEL if (USE_GEMINI and GEMINI_API_KEY) else None,
        "mode": TRADE_DECIDER,  # "rules" | "llm"
        "last_ts": None,
        "last_text": None
    },

    # failsafe
    "failsafe": {
        "enabled": True,
        "baseline_equity": None,
        "triggered": False,
        "triggered_at": None,
        "drawdown_pct": 0.0,
        "threshold_pct": DRAW_DOWN_LIMIT,
        "equity_now": None
    }
}

# ---------- helpers ----------
def _seconds_until(hour: int, minute: int, tz: ZoneInfo) -> int:
    now = datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(0, int((target - now).total_seconds()))

def _next_eod_dt(tz: ZoneInfo) -> datetime:
    now = datetime.now(tz)
    target = now.replace(hour=DAILY_EOD_HOUR, minute=DAILY_EOD_MINUTE, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target

def alpaca_symbol_no_slash(symbol: str) -> str:
    return symbol.replace("/", "")

def _save_open_trade(entry_time_iso: str, entry_price: float, qty: float, entry_signal: str):
    d = {"EntryTime": entry_time_iso, "EntryPrice": float(entry_price), "Qty": float(qty), "EntrySignal": entry_signal or "none"}
    with open(OPEN_TRADE, "w") as f:
        json.dump(d, f)
    state["entry_time"] = entry_time_iso
    state["entry_price"] = float(entry_price)
    state["entry_qty"]   = float(qty)

def _load_open_trade():
    if not OPEN_TRADE.exists():
        return None
    try:
        with open(OPEN_TRADE, "r") as f:
            return json.load(f)
    except Exception:
        return None

def _clear_open_trade():
    if OPEN_TRADE.exists():
        try: OPEN_TRADE.unlink()
        except: pass
    state["entry_time"] = None
    state["entry_price"] = None
    state["entry_qty"] = 0.0

def _read_baseline_json() -> float | None:
    if not BASELINE_JSON.exists():
        return None
    try:
        with open(BASELINE_JSON, "r") as f:
            d = json.load(f)
            val = float(d.get("baseline_equity", 0.0))
            return val if val > 0 else None
    except Exception:
        return None

def _write_baseline_json(new_base: float):
    with open(BASELINE_JSON, "w") as f:
        json.dump({"baseline_equity": float(new_base), "set_at": datetime.now(LOCAL_TZ).isoformat()}, f)

def _load_or_init_baseline() -> float:
    val = _read_baseline_json()
    if val is not None:
        return val
    cur = get_equity_usd()
    _write_baseline_json(cur)
    state["failsafe"]["baseline_equity"] = float(cur)
    return float(cur)

def get_position_details(symbol: str) -> dict:
    sym_noslash = alpaca_symbol_no_slash(symbol)
    for s in (sym_noslash, symbol):
        try:
            p = trading_client.get_open_position(s)
            qty = float(getattr(p, "qty", 0.0) or 0.0)
            avg = getattr(p, "avg_entry_price", None)
            avg = float(avg) if avg not in (None, "") else None
            return {"qty": qty, "avg_entry": avg}
        except Exception:
            pass
    try:
        positions = trading_client.get_all_positions()
        for p in positions:
            if getattr(p, "symbol", "") in (symbol, sym_noslash):
                qty = float(getattr(p, "qty", 0.0) or 0.0)
                avg = getattr(p, "avg_entry_price", None)
                avg = float(avg) if avg not in (None, "") else None
                return {"qty": qty, "avg_entry": avg}
    except Exception:
        pass
    return {"qty": 0.0, "avg_entry": None}

def refresh_state_from_broker():
    pos = get_position_details(SYMBOL)
    state["position_qty"] = pos["qty"]
    if pos["qty"] > 0:
        state["position_side"] = "long"
        if state["entry_price"] is None or state["entry_qty"] <= 0:
            ot = _load_open_trade()
            if ot:
                state["entry_time"] = ot.get("EntryTime")
                state["entry_price"] = float(ot.get("EntryPrice", pos["avg_entry"] or 0.0))
                state["entry_qty"]   = float(ot.get("Qty", pos["qty"]))
            else:
                now_iso = datetime.now(LOCAL_TZ).isoformat()
                _save_open_trade(now_iso, pos["avg_entry"] or 0.0, pos["qty"], entry_signal="unknown_seed")
        else:
            if not OPEN_TRADE.exists():
                _save_open_trade(state["entry_time"], state["entry_price"], state["entry_qty"], entry_signal="unknown_restore")
    else:
        state["position_side"] = "flat"
        state["entry_time"] = None
        state["entry_price"] = None
        state["entry_qty"] = 0.0

# Indicators/Signals
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

# Data
def _fetch_bars_once(sym: str, days_back: int = 120) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(days=days_back)
    req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Hour, start=start)
    df = data_client.get_crypto_bars(req).df
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(sym, level="symbol")
        except KeyError:
            pass
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def fetch_hourly_bars(symbol: str, _days_back: int = 120) -> pd.DataFrame:
    df = _fetch_bars_once(symbol, _days_back)
    if df.empty:
        df = _fetch_bars_once(symbol.replace("/", ""), _days_back)
    if df.empty:
        log.warning(f"[data] No bars returned for {symbol}")
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = df.tz_convert(LOCAL_TZ)
    return df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})

# Account / equity
def get_account_summary() -> dict:
    try:
        a = trading_client.get_account()
        return {
            "equity": float(getattr(a, "equity", 0.0) or 0.0),
            "cash": float(getattr(a, "cash", 0.0) or 0.0),
            "buying_power": float(getattr(a, "buying_power", 0.0) or 0.0),
            "portfolio_value": float(getattr(a, "portfolio_value", getattr(a, "equity", 0.0)) or 0.0),
            "currency": "USD",
        }
    except Exception as e:
        return {"error": str(e)}

def get_equity_usd() -> float:
    a = trading_client.get_account()
    return float(a.equity)

# >>> NEW — safer notional helper
def _floor2(x: float) -> float:
    return math.floor(x * 100.0) / 100.0

def _safe_buy_notional(equity_before: float) -> dict:
    """
    Compute a conservative notional:
      min(alloc, available, equity_before*(1-FEE_BUFFER)) * (1-NOTIONAL_SAFETY_PCT) - NOTIONAL_SAFETY_USD
      floored to 2 decimals and clamped >= 0.
    """
    try:
        a = trading_client.get_account()
        bp = float(getattr(a, "buying_power", 0.0) or 0.0)
        cash = float(getattr(a, "cash", 0.0) or 0.0)
    except Exception:
        bp = 0.0
        cash = 0.0
    available = max(0.0, min(bp, cash) if cash > 0 else bp)
    alloc = max(0.0, available * BUY_BP_PCT)

    target = min(alloc, available, max(0.0, equity_before * (1.0 - FEE_BUFFER_PCT)))
    # extra headroom to avoid broker-side rounding/fee rejects
    target = target * (1.0 - NOTIONAL_SAFETY_PCT) - NOTIONAL_SAFETY_USD
    target = _floor2(max(0.0, target))
    return {"available": available, "alloc": alloc, "notional": target}

def _looks_like_insufficient_balance(err_msg: str) -> bool:
    if not err_msg: return False
    s = err_msg.lower()
    return ("insufficient balance" in s) or ("insufficient funds" in s) or ("40310000" in s)

def _update_baseline_if_new_high(equity_now: float):
    baseline = state["failsafe"]["baseline_equity"]
    if baseline is None or baseline <= 0:
        baseline = _load_or_init_baseline()
        state["failsafe"]["baseline_equity"] = baseline
    if equity_now > baseline:
        _write_baseline_json(equity_now)
        state["failsafe"]["baseline_equity"] = equity_now
        log.info(f"[FAILSAFE] Baseline raised to new high: ${equity_now:.2f}")

def _failsafe_check_and_stop_if_needed():
    baseline = state["failsafe"]["baseline_equity"]
    if not baseline or baseline <= 0:
        baseline = _load_or_init_baseline()
        state["failsafe"]["baseline_equity"] = baseline

    equity_now = get_equity_usd()
    state["failsafe"]["equity_now"] = equity_now

    if not state["failsafe"]["triggered"]:
        _update_baseline_if_new_high(equity_now)
        baseline = state["failsafe"]["baseline_equity"]

    dd = 0.0 if baseline == 0 else max(0.0, (baseline - equity_now) / baseline)
    state["failsafe"]["drawdown_pct"] = dd

    if dd >= DRAW_DOWN_LIMIT and not state["failsafe"]["triggered"]:
        state["failsafe"]["triggered"] = True
        state["failsafe"]["triggered_at"] = datetime.now(LOCAL_TZ).isoformat()
        run_flag["running"] = False
        log.error(f"[FAILSAFE] Drawdown {dd:.2%} reached (threshold {DRAW_DOWN_LIMIT:.0%}). Trading stopped.")

# Orders
def _wait_for_fill(order_id: str, timeout_s: float = 30.0, poll_s: float = 0.5):
    from time import sleep, time as _now
    end = _now() + timeout_s
    filled_avg = None
    filled_qty = None
    status = None
    while _now() < end:
        try:
            o = trading_client.get_order_by_id(order_id)
            status = getattr(o, "status", "").lower()
            if status in ("filled", "partially_filled", "canceled", "rejected", "expired"):
                fpx = getattr(o, "filled_avg_price", None)
                fqty = getattr(o, "filled_qty", None)
                if fpx is not None:
                    try: filled_avg = float(fpx)
                    except: pass
                if fqty is not None:
                    try: filled_qty = float(fqty)
                    except: pass
                break
        except Exception:
            pass
        sleep(poll_s)
    return status, filled_avg, filled_qty

def place_market_buy_notional_and_wait(symbol: str, notional_usd: float):
    if notional_usd <= 0:
        return None, None, None, None
    req = MarketOrderRequest(
        symbol=symbol,
        notional=str(round(notional_usd, 2)),
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC
    )
    o = trading_client.submit_order(req)
    status, fpx, fqty = _wait_for_fill(getattr(o, "id", ""))
    return o, status, fpx, fqty

def place_market_sell_all_and_wait(symbol: str, qty: float):
    if qty <= 0:
        return None, None, None, None
    req = MarketOrderRequest(
        symbol=symbol,
        qty=str(qty),
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    o = trading_client.submit_order(req)
    status, fpx, fqty = _wait_for_fill(getattr(o, "id", ""))
    return o, status, fpx, fqty

# Persistence: errors & trades
def append_error(ts: datetime, code: str, message: str, context: dict):
    row = {"timestamp": ts.isoformat(), "code": code, "message": message, "context": str(context or {})}
    df = pd.DataFrame([row])
    header = not ERRORS_CSV.exists()
    df.to_csv(ERRORS_CSV, mode="a", index=False, header=header)

def _ensure_trades_csv_header():
    if not TRADES_CSV.exists():
        pd.DataFrame(columns=TRADELOG_COLS).to_csv(TRADES_CSV, index=False)

def append_roundtrip_trade(entry_time: str, entry_price: float, exit_time: str, exit_price: float,
                           qty: float, entry_signal: str, exit_signal: str, note: str):
    _ensure_trades_csv_header()
    entry_value = float(entry_price) * float(qty)
    exit_value  = float(exit_price)  * float(qty)
    pnl_realized = exit_value - entry_value
    row = {
        "EntryTime": entry_time,
        "EntryPrice": float(entry_price),
        "ExitTime": exit_time,
        "ExitPrice": float(exit_price),
        "Qty": float(qty),
        "EntryValue": entry_value,
        "ExitValue": exit_value,
        "PnL_Realized": pnl_realized,
        "EntrySignal": entry_signal or "none",
        "ExitSignal": exit_signal or "none",
        "Note": note or ""
    }
    pd.DataFrame([row]).to_csv(TRADES_CSV, mode="a", index=False, header=False)

def load_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        return pd.DataFrame(columns=TRADELOG_COLS)
    return pd.read_csv(TRADES_CSV)

def load_equity() -> pd.DataFrame:
    if not EQUITY_CSV.exists():
        return pd.DataFrame(columns=["timestamp","equity","price","regime"])
    return pd.read_csv(EQUITY_CSV)

def load_errors() -> pd.DataFrame:
    if not ERRORS_CSV.exists():
        return pd.DataFrame(columns=["timestamp","code","message","context"])
    return pd.read_csv(ERRORS_CSV)

# LLM
def llm_enabled() -> bool:
    return bool(USE_GEMINI and GEMINI_API_KEY)

def llm_decide_action(context: dict) -> tuple[str, str]:
    if not llm_enabled():
        return "hold", "LLM disabled"
    try:
        from google.generativeai import configure, GenerativeModel
        configure(api_key=GEMINI_API_KEY)
        prompt = f"""
You are a BTC 1h strategy decider. Choose BUY, SELL or HOLD.
Rules:
- If EMA250 > EMA500 (bull): act on EMA cross (up=BUY, down=SELL)
- Else: act on CCI(20) vs its SMA(20) cross (up=BUY, down=SELL)
- If already LONG, ignore BUY; if FLAT, ignore SELL; otherwise HOLD.
Context JSON:
{context}
Return exactly:
action: <BUY|SELL|HOLD>
rationale: <one short sentence>
"""
        model = GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        action, rationale = "HOLD", ""
        for line in text.splitlines():
            L = line.strip()
            if L.lower().startswith("action:"):
                action = L.split(":",1)[1].strip().upper()
            elif L.lower().startswith("rationale:"):
                rationale = L.split(":",1)[1].strip()
        if action not in ("BUY","SELL","HOLD"):
            return "hold", f"Invalid LLM action: {action}"
        return action.lower(), (rationale or "—")
    except Exception as e:
        return "hold", f"LLM error: {e}"

# EOD helpers (single-send lock)
def _daily_lock_path(date_obj) -> Path:
    d = str(date_obj)
    return DATA_DIR / f"eod_sent_{d}.lock"

def eod_sent_today() -> bool:
    return _daily_lock_path(datetime.now(LOCAL_TZ).date()).exists()

def eod_try_acquire_lock_for_today() -> bool:
    today = datetime.now(LOCAL_TZ).date()
    path = _daily_lock_path(today)
    try:
        with open(path, "x") as f:
            f.write(datetime.now(LOCAL_TZ).isoformat())
        return True
    except FileExistsError:
        return False
    except Exception as e:
        log.warning(f"EOD lock error ({path}): {e}")
        return False

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
        msg.add_attachment(data, maintype="application", subtype="octet-stream", filename=path.name)
    with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
        smtp.starttls()
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.send_message(msg)

def send_eod_email_now(force: bool = False) -> dict:
    try:
        if (not force) and (not eod_try_acquire_lock_for_today()):
            return {"skipped": True, "reason": "already_sent_today"}
        now = datetime.now(LOCAL_TZ)
        trades = load_trades()
        errors = load_errors()

        xlsx_path = DATA_DIR / f"eod_{now.date()}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as xw:
            (trades if not trades.empty else pd.DataFrame(columns=TRADELOG_COLS)).to_excel(
                xw, index=False, sheet_name="Trades"
            )
            (errors if not errors.empty else pd.DataFrame(columns=[
                "timestamp","code","message","context"
            ])).to_excel(xw, index=False, sheet_name="Errors")

        daily_pnl = 0.0
        overall_pnl = 0.0
        if not trades.empty:
            trades["EntryTime"] = pd.to_datetime(trades["EntryTime"], errors="coerce")
            trades["ExitTime"]  = pd.to_datetime(trades["ExitTime"], errors="coerce")
            trades["PnL_Realized"] = pd.to_numeric(trades["PnL_Realized"], errors="coerce").fillna(0.0)
            today = now.date()
            daily_pnl = float(trades[trades["ExitTime"].dt.date == today]["PnL_Realized"].sum())
            overall_pnl = float(trades["PnL_Realized"].sum())

        subject = f"[BTC Bot] EOD {now.date()} — Daily: {daily_pnl:+.2f} | Overall: {overall_pnl:+.2f}"
        body = (
            f"Symbol: {SYMBOL}\n"
            f"Date: {now.date()}\n"
            f"Daily Realized P&L: {daily_pnl:+.2f}\n"
            f"Overall Realized P&L: {overall_pnl:+.2f}\n"
            f"Running status: {'RUNNING' if run_flag['running'] else 'STOPPED'}\n"
            f"Regime: {state.get('regime')}\n"
            f"Last signal/action: {state.get('last_signal')}/{state.get('last_action')}\n"
            f"Decider: {state.get('llm',{}).get('mode')}\n"
            f"\nAttached: eod_{now.date()}.xlsx (Trades & Errors)\n"
        )
        send_email(subject, body, attachments=[xlsx_path])
        return {"message": "EOD email sent", "subject": subject, "xlsx": str(xlsx_path)}
    except Exception as e:
        log.exception("send_eod_email_now failed")
        return {"error": str(e)}

def eod_scheduler_loop():
    log.info(f"EOD scheduler started (daily at {DAILY_EOD_HOUR:02d}:{DAILY_EOD_MINUTE:02d} {LOCAL_TZ})")
    while True:
        try:
            sleep_s = _seconds_until(DAILY_EOD_HOUR, DAILY_EOD_MINUTE, LOCAL_TZ)
        except Exception:
            sleep_s = 3600
        time.sleep(max(5, sleep_s))
        res = send_eod_email_now(force=False)
        log.info(f"[EOD] scheduler result: {res}")

threading.Thread(target=eod_scheduler_loop, daemon=True).start()

# ---------- core eval/trade ----------
def evaluate_and_trade():
    try:
        # Failsafe before anything
        _failsafe_check_and_stop_if_needed()
        if state["failsafe"]["triggered"]:
            return

        bars = fetch_hourly_bars(SYMBOL, LOOKBACK_BARS)
        if bars.empty:
            state["last_error"] = "No bars from data API."
            return
        if len(bars) < 510:
            state["last_error"] = "Insufficient bars for EMA(500)."
            return

        last_time = bars.index[-1]
        if state["last_bar_time"] == last_time:
            return

        close = bars["close"]
        price = float(close.iloc[-1])
        ema250 = ema(close, 250); ema500 = ema(close, 500)
        dfhlc = bars[["high","low","close"]].copy()
        _cci = cci(dfhlc, 20); _cci_sma = cci_sma(_cci, 20)

        e250_prev, e500_prev = float(ema250.iloc[-2]), float(ema500.iloc[-2])
        e250_curr, e500_curr = float(ema250.iloc[-1]), float(ema500.iloc[-1])
        cci_prev, cci_sma_prev = float(_cci.iloc[-2]), float(_cci_sma.iloc[-2])
        cci_curr, cci_sma_curr = float(_cci.iloc[-1]), float(_cci_sma.iloc[-1])

        bull = e250_curr > e500_curr
        ema_cross_up   = crossed_above(e250_prev, e250_curr, e500_prev, e500_curr)
        ema_cross_down = crossed_below(e250_prev, e250_curr, e500_prev, e500_curr)
        cci_cross_up   = crossed_above(cci_prev, cci_curr, cci_sma_prev, cci_sma_curr)
        cci_cross_down = crossed_below(cci_prev, cci_curr, cci_sma_prev, cci_sma_curr)

        # rules suggestion
        rules_action = "hold"; signal = "none"
        if bull:
            if ema_cross_up:     signal, rules_action = "ema_cross_up", "buy"
            elif ema_cross_down: signal, rules_action = "ema_cross_down", "sell"
        else:
            if cci_cross_up:     signal, rules_action = "cci_cross_up", "buy"
            elif cci_cross_down: signal, rules_action = "cci_cross_down", "sell"

        # choose decider
        decider = state["llm"]["mode"]
        action = rules_action
        if decider == "llm":
            ctx = dict(
                symbol=SYMBOL, regime=("bull" if bull else "nonbull"),
                ema250_prev=e250_prev, ema250_curr=e250_curr,
                ema500_prev=e500_prev, ema500_curr=e500_curr,
                ema_cross_up=ema_cross_up, ema_cross_down=ema_cross_down,
                cci_prev=cci_prev, cci_curr=cci_curr,
                cci_sma_prev=cci_sma_prev, cci_sma_curr=cci_sma_curr,
                cci_cross_up=cci_cross_up, cci_cross_down=cci_cross_down,
                position=("long" if get_position_details(SYMBOL)["qty"] > 0 else "flat"),
                rules_suggestion=rules_action
            )
            act, rationale = llm_decide_action(ctx)
            qty_now_tmp = get_position_details(SYMBOL)["qty"]
            if act == "buy" and qty_now_tmp > 0:  act = "hold"
            if act == "sell" and qty_now_tmp == 0: act = "hold"
            action = act
            state["llm"]["last_text"] = rationale or "—"
            state["llm"]["last_ts"] = datetime.now(LOCAL_TZ).isoformat()
        else:
            state["llm"]["last_text"] = f"Rules mode: {signal or 'none'} → {rules_action}"
            state["llm"]["last_ts"] = datetime.now(LOCAL_TZ).isoformat()

        # snapshot before order
        pos_before = get_position_details(SYMBOL)
        qty_now = pos_before["qty"]
        equity_before = get_equity_usd()

        if action == "buy" and qty_now <= 0:
            # >>> NEW — safe notional + retries against insufficient balance
            sizing = _safe_buy_notional(equity_before)
            notional = sizing["notional"]
            if notional < MIN_NOTIONAL_USD:
                msg = (f"Insufficient funds after safety: available≈{sizing['available']:.2f}, "
                       f"alloc≈{sizing['alloc']:.2f}, safe_notional≈{notional:.2f}, "
                       f"required_min≈{MIN_NOTIONAL_USD:.2f}")
                append_error(
                    ts=last_time,
                    code="insufficient_funds_safety",
                    message=msg,
                    context={"action":"buy","signal":signal,"price":price,"equity_before":equity_before,
                             "available":sizing["available"],"alloc":sizing["alloc"],"safe_notional":notional,
                             "min_notional":MIN_NOTIONAL_USD}
                )
                state["last_error"] = msg
            else:
                attempt = 0
                curr_notional = notional
                last_err_msg = None
                while attempt < RETRY_MAX_ATTEMPTS:
                    try:
                        order_resp, status, fpx, fqty = place_market_buy_notional_and_wait(SYMBOL, curr_notional)
                        if status in ("rejected", "canceled", "expired"):
                            # broker-side non-fillable at this notional; back off
                            last_err_msg = f"Buy {status} at notional={curr_notional:.2f}"
                            append_error(ts=last_time, code="order_rejected",
                                         message=last_err_msg,
                                         context={"action":"buy","signal":signal,"price":price,"status":status,"notional":curr_notional,"attempt":attempt+1})
                            # back off and retry
                            curr_notional = _floor2(max(MIN_NOTIONAL_USD, curr_notional * (1.0 - RETRY_BACKOFF_PCT) - NOTIONAL_SAFETY_USD))
                            attempt += 1
                            continue

                        # success path
                        filled_px = fpx if fpx else price
                        filled_qty = fqty if (fqty and fqty > 0) else round(curr_notional / price, 8)
                        entry_time_iso = last_time.isoformat()
                        _save_open_trade(entry_time_iso, filled_px, filled_qty, entry_signal=signal)
                        state["position_side"] = "long"
                        last_err_msg = None
                        break

                    except Exception as e:
                        emsg = str(e)
                        last_err_msg = emsg
                        append_error(ts=last_time, code="order_exception_retry",
                                     message=f"Buy error on attempt {attempt+1}: {emsg}",
                                     context={"action":"buy","signal":signal,"price":price,"notional":curr_notional,"attempt":attempt+1})
                        if _looks_like_insufficient_balance(emsg):
                            # back off further and retry
                            curr_notional = _floor2(max(MIN_NOTIONAL_USD, curr_notional * (1.0 - RETRY_BACKOFF_PCT) - NOTIONAL_SAFETY_USD))
                            attempt += 1
                            continue
                        else:
                            # non-balance error: stop retrying
                            break

                if last_err_msg:
                    state["last_error"] = last_err_msg

        elif action == "sell" and qty_now > 0:
            try:
                sell_qty = qty_now
                order_resp, status, fpx, fqty = place_market_sell_all_and_wait(SYMBOL, sell_qty)
                filled_px = fpx if fpx else price
                filled_qty = fqty if (fqty and fqty > 0) else sell_qty

                ot = _load_open_trade() or {}
                entry_time_iso = ot.get("EntryTime") or state.get("entry_time") or last_time.isoformat()
                entry_price    = float(ot.get("EntryPrice", state.get("entry_price") or filled_px))
                entry_qty      = float(ot.get("Qty", state.get("entry_qty") or filled_qty))
                entry_signal   = ot.get("EntrySignal", state.get("last_signal") or "none")

                append_roundtrip_trade(
                    entry_time=entry_time_iso,
                    entry_price=entry_price,
                    exit_time=last_time.isoformat(),
                    exit_price=filled_px,
                    qty=entry_qty,
                    entry_signal=entry_signal,
                    exit_signal=signal,
                    note=f"order_id={getattr(order_resp,'id','')} | decider={state['llm']['mode']}"
                )

                _clear_open_trade()
                state["position_side"] = "flat"

                if status in ("rejected", "canceled", "expired"):
                    append_error(
                        ts=last_time, code="order_rejected",
                        message=f"Sell {status} by broker",
                        context={"action":"sell","signal":signal,"price":price,"status":status,"qty":sell_qty}
                    )
                    state["last_error"] = f"Sell {status}"
            except Exception as e:
                msg = f"Sell error: {e}"
                append_error(
                    ts=last_time, code="order_exception",
                    message=msg,
                    context={"action":"sell","signal":signal,"price":price,"qty":qty_now}
                )
                state["last_error"] = msg

        # indicators to UI
        state["indicators"] = {
            "last_bar_time": str(last_time),
            "price": price,
            "ema250_prev": float(ema250.iloc[-2]),
            "ema250_curr": float(ema250.iloc[-1]),
            "ema500_prev": float(ema500.iloc[-2]),
            "ema500_curr": float(ema500.iloc[-1]),
            "ema_spread_curr": float(ema250.iloc[-1] - ema500.iloc[-1]),
            "ema_cross_up": bool(ema_cross_up),
            "ema_cross_down": bool(ema_cross_down),
            "cci_prev": float(_cci.iloc[-2]),
            "cci_curr": float(_cci.iloc[-1]),
            "cci_sma_prev": float(_cci_sma.iloc[-2]),
            "cci_sma_curr": float(_cci_sma.iloc[-1]),
            "cci_diff_curr": float(_cci.iloc[-1] - _cci_sma.iloc[-1]),
            "cci_cross_up": bool(cci_cross_up),
            "cci_cross_down": bool(cci_cross_down),
        }

        state["last_bar_time"] = last_time
        refresh_state_from_broker()
        state["last_signal"] = signal
        state["last_action"] = action
        state["regime"]      = "bull" if (ema250.iloc[-1] > ema500.iloc[-1]) else "nonbull"
        state["updated_at"]  = datetime.now(LOCAL_TZ).isoformat()

        # equity snapshot
        equity_after = get_equity_usd()
        row = {"timestamp": last_time.isoformat(), "equity": equity_after, "price": price, "regime": state["regime"]}
        pd.DataFrame([row]).to_csv(EQUITY_CSV, mode=("a" if EQUITY_CSV.exists() else "w"), index=False, header=not EQUITY_CSV.exists())

    except Exception as e:
        log.exception("evaluate_and_trade error")
        state["last_error"] = str(e)
        state["updated_at"] = datetime.now(LOCAL_TZ).isoformat()

# ---------- worker ----------
def worker_loop():
    log.info("Worker started.")
    try:
        baseline = _load_or_init_baseline()
        state["failsafe"]["baseline_equity"] = baseline
        refresh_state_from_broker()
    except Exception as e:
        state["last_error"] = str(e)

    while run_flag["running"]:
        _failsafe_check_and_stop_if_needed()
        if not run_flag["running"]:
            break
        evaluate_and_trade()
        time.sleep(SLEEP_SEC)

    log.info("Worker stopped.")

# ---------- routes ----------
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
    s["account"] = get_account_summary()
    s["account_at"] = datetime.now(LOCAL_TZ).isoformat()
    s["eod_next_at"] = _next_eod_dt(LOCAL_TZ).isoformat()
    s["eod_sent_today"] = eod_sent_today()
    s["symbol"] = SYMBOL
    return jsonify(s)

@app.route("/trades.json")
def trades_json():
    try:
        df = load_trades()
        df = df.tail(500) if not df.empty else df
        data = df.to_json(orient="records")
        resp = make_response(data, 200)
        resp.mimetype = "application/json"
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp
    except Exception as e:
        app.logger.exception("/trades.json failed")
        return jsonify(error=str(e)), 500

@app.route("/equity.json")
def equity_json():
    df = load_equity().tail(1000)
    return df.to_json(orient="records")

@app.route("/download/trades.csv")
def download_trades_csv():
    if not TRADES_CSV.exists():
        return jsonify(error="No trades yet"), 404
    return send_file(TRADES_CSV, as_attachment=True, download_name="trades.csv")

@app.route("/download/errors.csv")
def download_errors_csv():
    if not ERRORS_CSV.exists():
        return jsonify(error="No errors logged"), 404
    return send_file(ERRORS_CSV, as_attachment=True, download_name="errors.csv")

@app.route("/start", methods=["POST","GET"])
def start():
    if run_flag["running"]:
        return jsonify(message="already running"), 200
    try:
        baseline = _load_or_init_baseline()
        state["failsafe"]["baseline_equity"] = baseline
    except Exception as e:
        state["last_error"] = str(e)
    run_flag["running"] = True
    threading.Thread(target=worker_loop, daemon=True).start()
    return jsonify(message="started"), 200

@app.route("/stop", methods=["POST","GET"])
def stop():
    if not run_flag["running"]:
        return jsonify(message="already stopped"), 200
    run_flag["running"] = False
    return jsonify(message="stopping"), 200

# ---- Failsafe reset control ----
@app.route("/failsafe/reset", methods=["POST"])
def failsafe_reset():
    """
    Sets baseline_equity = current equity, clears trigger.
    Does NOT auto-start trading.
    """
    try:
        cur = get_equity_usd()
        _write_baseline_json(cur)
        state["failsafe"]["baseline_equity"] = cur
        state["failsafe"]["triggered"] = False
        state["failsafe"]["triggered_at"] = None
        state["failsafe"]["drawdown_pct"] = 0.0
        state["failsafe"]["equity_now"] = cur
        return jsonify(message="Failsafe reset", baseline=cur), 200
    except Exception as e:
        return jsonify(error=str(e)), 500

# ---- LLM endpoints ----
@app.route("/config/decider", methods=["GET","POST"])
def config_decider():
    mode = None
    if request.method == "POST":
        mode = request.args.get("mode")
        if request.is_json and not mode:
            body = request.get_json(silent=True) or {}
            mode = body.get("mode")
    if mode:
        m = str(mode).strip().lower()
        if m not in ("rules","llm"):
            return jsonify(error="mode must be 'rules' or 'llm'"), 400
        state["llm"]["mode"] = m
        return jsonify(message=f"decider set to {m}", mode=m), 200
    return jsonify(mode=state["llm"]["mode"], enabled=llm_enabled(), model=state["llm"]["model"])

# ---- EOD endpoint ----
@app.route("/debug/send-eod", methods=["POST","GET"])
def debug_send_eod():
    force = request.args.get("force", "0") in ("1", "true", "True")
    res = send_eod_email_now(force=force)
    code = 200 if "error" not in res else 500
    return jsonify(res), code

# ---------- entry ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)