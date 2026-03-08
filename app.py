"""
Options Chain Dashboard — Flask Web App
CBOE-style dark theme, yfinance data, Supabase auth + snapshots
"""

import math
import time
import os
from datetime import datetime, timedelta
from functools import wraps

import requests

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, session,
    redirect, url_for, jsonify
)
from supabase import create_client

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
ADMIN_KEY    = os.environ.get("ADMIN_KEY", "")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

INDICES   = ["SPY", "QQQ", "IWM", "DIA"]
MEGA_CAPS = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
FRED_VIX_ID  = "VIXCLS"

CBOE_SENT_TICKERS = {
    "SKEW":  ("^SKEW",  "CBOE SKEW Index — OTM put demand vs ATM (130+ = heavy hedging)"),
    "VIX3M": ("^VIX3M", "CBOE 3-Month VIX — longer-term fear gauge"),
    "VIX9D": ("^VIX9D", "CBOE 9-Day VIX — very short-term implied vol"),
    "VIX6M": ("^VIX6M", "CBOE 6-Month VIX — intermediate-term fear"),
}

# ─────────────────────────── In-Memory TTL Cache ───────────────────────────

_cache: dict = {}
CACHE_TTL = 300  # 5 minutes


def _cache_get(key):
    entry = _cache.get(key)
    if entry is None:
        return None
    data, ts = entry
    if time.time() - ts > CACHE_TTL:
        del _cache[key]
        return None
    return data


def _cache_set(key, data):
    _cache[key] = (data, time.time())


# ─────────────────────────── Auth Helpers ───────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            # Return JSON for API routes, redirect for page routes
            if request.path.startswith("/api/"):
                return jsonify({"error": "Session expired — please log in again"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-Admin-Key") or request.json.get("admin_key", "") if request.is_json else request.headers.get("X-Admin-Key", "")
        if not ADMIN_KEY or key != ADMIN_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────── Black-Scholes Greeks ───────────────────────────

def bs_greeks(S, K, iv_pct, expiry_str, option_type):
    """Compute delta, gamma, theta via Black-Scholes (no scipy)."""
    try:
        iv = iv_pct / 100.0
        if iv <= 0 or S <= 0 or K <= 0:
            return None, None, None
        exp_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        T = max((exp_date - datetime.now()).days / 365.0, 1 / 365.0)
        r  = 0.05
        d1 = (math.log(S / K) + (r + 0.5 * iv ** 2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)

        def N(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        def n(x):
            return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

        if option_type == "calls":
            delta = N(d1)
            theta = ((-S * n(d1) * iv / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * N(d2)) / 365
        else:
            delta = N(d1) - 1
            theta = ((-S * n(d1) * iv / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * N(-d2)) / 365
        gamma = n(d1) / (S * iv * math.sqrt(T))
        return round(delta, 4), round(gamma, 6), round(theta, 4)
    except Exception:
        return None, None, None


# ─────────────────────────── Analysis Functions ───────────────────────────

def calc_max_pain(calls_df, puts_df):
    all_strikes = sorted(set(calls_df["strike"]).union(set(puts_df["strike"])))
    if not all_strikes:
        return None
    min_pain = float("inf")
    max_pain_strike = all_strikes[0]
    for s in all_strikes:
        call_pain = float(((calls_df["strike"] - s).clip(lower=0) * calls_df["openInterest"]).sum())
        put_pain  = float(((s - puts_df["strike"]).clip(lower=0)  * puts_df["openInterest"]).sum())
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = s
    return float(max_pain_strike)


def calc_iv_skew(calls_df, puts_df, price):
    if price is None:
        return None
    otm_puts  = puts_df[puts_df["strike"]   < price]
    otm_calls = calls_df[calls_df["strike"] > price]
    if otm_puts.empty or otm_calls.empty:
        return None
    put_iv  = otm_puts["impliedVolatility"].replace(0, float("nan")).mean()
    call_iv = otm_calls["impliedVolatility"].replace(0, float("nan")).mean()
    if pd.isna(put_iv) or pd.isna(call_iv):
        return None
    return round(float(put_iv - call_iv), 2)


def calc_expected_move(calls_df, puts_df, price):
    if price is None:
        return None
    all_strikes = set(calls_df["strike"]).union(set(puts_df["strike"]))
    if not all_strikes:
        return None
    atm = min(all_strikes, key=lambda s: abs(s - price))
    atm_call = calls_df[calls_df["strike"] == atm]["lastPrice"]
    atm_put  = puts_df[puts_df["strike"]   == atm]["lastPrice"]
    if atm_call.empty or atm_put.empty:
        return None
    return round(float(atm_call.iloc[0]) + float(atm_put.iloc[0]), 2)


def calc_walls(calls_df, puts_df):
    call_wall = put_wall = None
    if not calls_df.empty and calls_df["openInterest"].sum() > 0:
        call_wall = float(calls_df.loc[calls_df["openInterest"].idxmax(), "strike"])
    if not puts_df.empty and puts_df["openInterest"].sum() > 0:
        put_wall  = float(puts_df.loc[puts_df["openInterest"].idxmax(),  "strike"])
    return put_wall, call_wall


def calc_gex(calls_df, puts_df, price):
    if price is None:
        return None
    S = price
    total = 0.0
    for df, sign in [(calls_df, 1), (puts_df, -1)]:
        for _, row in df.iterrows():
            K  = row["strike"]
            iv = row["impliedVolatility"] / 100.0
            oi = row["openInterest"]
            if iv <= 0 or oi <= 0:
                continue
            try:
                T  = 1 / 52
                d1 = (math.log(S / K) + 0.5 * iv ** 2 * T) / (iv * math.sqrt(T))
                gamma = math.exp(-0.5 * d1 ** 2) / (math.sqrt(2 * math.pi) * S * iv * math.sqrt(T))
                total += sign * gamma * oi * 100 * S
            except Exception:
                continue
    return round(total / 1_000_000, 2)


def calc_iv_rank(calls_df, puts_df):
    all_iv = pd.concat([calls_df["impliedVolatility"], puts_df["impliedVolatility"]])
    all_iv = all_iv[all_iv > 0]
    if all_iv.empty:
        return None
    lo, hi, curr = float(all_iv.min()), float(all_iv.max()), float(all_iv.median())
    if hi == lo:
        return 50.0
    return round(100 * (curr - lo) / (hi - lo), 1)


def sentiment(pcr):
    if pcr < 0.7:
        return "BULLISH"
    if pcr > 1.0:
        return "BEARISH"
    return "NEUTRAL"


def clean_chain_df(df):
    needed = ["strike", "lastPrice", "bid", "ask", "volume",
              "openInterest", "impliedVolatility", "inTheMoney"]
    for col in needed:
        if col not in df.columns:
            df[col] = 0
    df = df[needed].copy()
    df.fillna(0, inplace=True)
    df["volume"]       = df["volume"].astype(int)
    df["openInterest"] = df["openInterest"].astype(int)
    df["impliedVolatility"] = (df["impliedVolatility"] * 100).round(2)
    return df


def build_chain_rows(df, side, price, expiry):
    rows = []
    atm_strike = None
    if price:
        atm_strike = float(df.iloc[(df["strike"] - price).abs().argsort()[:1]]["strike"].values[0]) if not df.empty else None
    oi_threshold = float(df["openInterest"].quantile(0.90)) if not df.empty else 0

    for _, row in df.iterrows():
        strike  = float(row["strike"])
        iv_pct  = float(row["impliedVolatility"])
        itm     = bool(row["inTheMoney"])
        oi      = int(row["openInterest"])
        vol     = int(row["volume"])

        if price and expiry and iv_pct > 0:
            delta, gamma, theta = bs_greeks(price, strike, iv_pct, expiry, side)
        else:
            delta = gamma = theta = None

        if atm_strike and abs(strike - atm_strike) < 0.01:
            row_class = "atm"
        elif itm:
            row_class = "itm" if side == "calls" else "itm_put"
        else:
            row_class = "otm"
        if oi >= oi_threshold > 0:
            row_class = "high_oi"
        if oi > 0 and vol >= oi * 3:
            row_class = "unusual_vol"

        rows.append({
            "strike":    strike,
            "lastPrice": float(row["lastPrice"]),
            "bid":       float(row["bid"]),
            "ask":       float(row["ask"]),
            "volume":    vol,
            "openInterest": oi,
            "iv":        iv_pct,
            "delta":     delta,
            "gamma":     gamma,
            "theta":     theta,
            "inTheMoney": itm,
            "rowClass":  row_class,
        })
    return rows


def _fetch_one_index(ticker):
    try:
        tk_obj      = yf.Ticker(ticker)
        expirations = tk_obj.options
        if not expirations:
            return ticker, None
        chain       = tk_obj.option_chain(expirations[0])
        calls, puts = chain.calls.copy(), chain.puts.copy()

        call_vol = int(calls["volume"].fillna(0).sum())
        put_vol  = int(puts["volume"].fillna(0).sum())
        call_oi  = int(calls["openInterest"].fillna(0).sum())
        put_oi   = int(puts["openInterest"].fillna(0).sum())
        vol_pcr  = round(put_vol / call_vol, 3) if call_vol > 0 else 0.0
        oi_pcr   = round(put_oi  / call_oi,  3) if call_oi  > 0 else 0.0

        try:
            price = float(tk_obj.fast_info.last_price)
        except Exception:
            price = None

        tiers = {}
        for df, side in [(calls, "CALLS"), (puts, "PUTS")]:
            lp  = df["lastPrice"].fillna(0)
            vol = df["volume"].fillna(0)
            lotto   = int(vol[lp < 1].sum())
            mid     = int(vol[(lp >= 1) & (lp < 5)].sum())
            premium = int(vol[lp >= 5].sum())
            total   = lotto + mid + premium
            tiers[side] = {
                "lotto":     lotto,
                "mid":       mid,
                "premium":   premium,
                "total":     total,
                "pct_lotto": round(100 * lotto   / total, 1) if total > 0 else 0.0,
                "pct_prem":  round(100 * premium / total, 1) if total > 0 else 0.0,
            }

        return ticker, {
            "call_vol":  call_vol,
            "put_vol":   put_vol,
            "call_oi":   call_oi,
            "put_oi":    put_oi,
            "vol_pcr":   vol_pcr,
            "oi_pcr":    oi_pcr,
            "price":     price,
            "tiers":     tiers,
            "sentiment": sentiment(vol_pcr),
        }
    except Exception:
        return ticker, None


def fetch_market_data_live():
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_one_index, t): t for t in INDICES}
        for fut in concurrent.futures.as_completed(futures):
            ticker, data = fut.result()
            if data:
                results[ticker] = data
    return results


def _fetch_one_mega(ticker):
    try:
        tk_obj      = yf.Ticker(ticker)
        expirations = tk_obj.options
        if not expirations:
            return ticker, None
        chain       = tk_obj.option_chain(expirations[0])
        calls, puts = chain.calls.copy(), chain.puts.copy()
        for df in [calls, puts]:
            df.fillna(0, inplace=True)
            df["volume"]       = df["volume"].astype(int)
            df["openInterest"] = df["openInterest"].astype(int)

        call_vol = int(calls["volume"].sum())
        put_vol  = int(puts["volume"].sum())
        pcr      = round(put_vol / call_vol, 3) if call_vol > 0 else 0.0

        try:
            price = float(tk_obj.fast_info.last_price)
        except Exception:
            price = None

        tiers = {}
        for df, side in [(calls, "calls"), (puts, "puts")]:
            lp  = df["lastPrice"]
            vol = df["volume"]
            l   = int(vol[lp < 1].sum())
            m   = int(vol[(lp >= 1) & (lp < 5)].sum())
            p   = int(vol[lp >= 5].sum())
            tiers[side] = {"lotto": l, "mid": m, "premium": p, "total": l+m+p}

        tc  = tiers.get("calls", {})
        tp  = tiers.get("puts",  {})
        tot = (tc.get("total", 0) + tp.get("total", 0)) or 1
        return ticker, {
            "call_vol":  call_vol,
            "put_vol":   put_vol,
            "pcr":       pcr,
            "price":     price,
            "tiers":     tiers,
            "pct_lotto": round(100 * (tc.get("lotto",0)+tp.get("lotto",0)) / tot, 1),
            "pct_prem":  round(100 * (tc.get("premium",0)+tp.get("premium",0)) / tot, 1),
            "sentiment": sentiment(pcr),
        }
    except Exception:
        return ticker, None


def fetch_mega_data_live():
    import concurrent.futures
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
        futures = {ex.submit(_fetch_one_mega, t): t for t in MEGA_CAPS}
        for fut in concurrent.futures.as_completed(futures):
            ticker, data = fut.result()
            if data:
                results[ticker] = data
    return results


# ─────────────────────────── Routes ───────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html", error=None)


@app.route("/login", methods=["POST"])
def login_post():
    code = request.form.get("invite_code", "").strip()
    if not code:
        return render_template("login.html", error="Please enter an invite code.")

    if supabase is None:
        return render_template("login.html", error="Auth service unavailable.")

    try:
        result = (
            supabase.table("subscribers")
            .select("id, active, tier")
            .eq("invite_code", code)
            .eq("active", True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            session["authenticated"] = True
            session["subscriber_id"] = row["id"]
            session["tier"]          = row.get("tier", "basic")
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid or inactive invite code.")
    except Exception as e:
        return render_template("login.html", error=f"Auth error: {e}")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────── API: Expiries ───────────────────────────

@app.route("/api/expiries")
@login_required
def api_expiries():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    try:
        tk_obj      = yf.Ticker(ticker)
        expirations = list(tk_obj.options)
        if not expirations:
            return jsonify({"error": f"No options found for {ticker}"}), 404
        return jsonify({"ticker": ticker, "expiries": expirations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: Chain ───────────────────────────

@app.route("/api/chain")
@login_required
def api_chain():
    ticker = request.args.get("ticker", "").strip().upper()
    expiry = request.args.get("expiry", "").strip()
    if not ticker or not expiry:
        return jsonify({"error": "ticker and expiry required"}), 400

    try:
        tk_obj = yf.Ticker(ticker)

        try:
            price = float(tk_obj.fast_info.last_price)
        except Exception:
            price = None

        chain      = tk_obj.option_chain(expiry)
        calls_raw  = clean_chain_df(chain.calls.copy())
        puts_raw   = clean_chain_df(chain.puts.copy())

        call_vol = int(calls_raw["volume"].sum())
        put_vol  = int(puts_raw["volume"].sum())
        call_oi  = int(calls_raw["openInterest"].sum())
        put_oi   = int(puts_raw["openInterest"].sum())
        vol_pcr  = round(put_vol / call_vol, 3) if call_vol > 0 else 0.0
        oi_pcr   = round(put_oi  / call_oi,  3) if call_oi  > 0 else 0.0

        max_pain   = calc_max_pain(calls_raw, puts_raw)
        iv_skew    = calc_iv_skew(calls_raw, puts_raw, price)
        exp_move   = calc_expected_move(calls_raw, puts_raw, price)
        put_wall, call_wall = calc_walls(calls_raw, puts_raw)
        gex        = calc_gex(calls_raw, puts_raw, price)
        iv_rank    = calc_iv_rank(calls_raw, puts_raw)

        call_rows = build_chain_rows(calls_raw, "calls", price, expiry)
        put_rows  = build_chain_rows(puts_raw,  "puts",  price, expiry)

        payload = {
            "ticker":    ticker,
            "expiry":    expiry,
            "price":     price,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "call_vol":  call_vol,
                "put_vol":   put_vol,
                "call_oi":   call_oi,
                "put_oi":    put_oi,
                "vol_pcr":   vol_pcr,
                "oi_pcr":    oi_pcr,
                "sentiment_vol": sentiment(vol_pcr),
                "sentiment_oi":  sentiment(oi_pcr),
            },
            "analysis": {
                "max_pain":  max_pain,
                "iv_skew":   iv_skew,
                "exp_move":  exp_move,
                "put_wall":  put_wall,
                "call_wall": call_wall,
                "gex":       gex,
                "iv_rank":   iv_rank,
            },
            "calls": call_rows,
            "puts":  put_rows,
        }

        # Save snapshot to Supabase (delete+insert pattern for upsert)
        if supabase:
            try:
                supabase.table("chain_snapshots").delete().eq("ticker", ticker).eq("expiry", expiry).execute()
                supabase.table("chain_snapshots").insert({
                    "ticker": ticker,
                    "expiry": expiry,
                    "data":   payload,
                }).execute()
            except Exception:
                pass

        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: Market ───────────────────────────

@app.route("/api/market")
@login_required
def api_market():
    cached = _cache_get("market")
    if cached:
        return jsonify(cached)

    try:
        data = fetch_market_data_live()

        total_cv = sum(d["call_vol"] for d in data.values())
        total_pv = sum(d["put_vol"]  for d in data.values())
        total_co = sum(d["call_oi"]  for d in data.values())
        total_po = sum(d["put_oi"]   for d in data.values())
        vol_pcr  = round(total_pv / total_cv, 3) if total_cv > 0 else 0.0
        oi_pcr   = round(total_po / total_co, 3) if total_co > 0 else 0.0

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "indices":   data,
            "aggregate": {
                "call_vol":      total_cv,
                "put_vol":       total_pv,
                "call_oi":       total_co,
                "put_oi":        total_po,
                "vol_pcr":       vol_pcr,
                "oi_pcr":        oi_pcr,
                "sentiment_vol": sentiment(vol_pcr),
                "sentiment_oi":  sentiment(oi_pcr),
            },
        }

        _cache_set("market", payload)

        if supabase:
            try:
                supabase.table("market_snapshots").insert({"data": payload}).execute()
            except Exception:
                pass

        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: Mega ───────────────────────────

@app.route("/api/mega")
@login_required
def api_mega():
    cached = _cache_get("mega")
    if cached:
        return jsonify(cached)

    try:
        data = fetch_mega_data_live()

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "tickers":   data,
        }

        _cache_set("mega", payload)

        if supabase:
            try:
                supabase.table("mega_snapshots").insert({"data": payload}).execute()
            except Exception:
                pass

        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: Snapshots ───────────────────────────

@app.route("/api/snapshot/chain")
@login_required
def api_snapshot_chain():
    ticker = request.args.get("ticker", "").strip().upper()
    expiry = request.args.get("expiry", "").strip()
    if not ticker or not expiry:
        return jsonify({"error": "ticker and expiry required"}), 400
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 503
    try:
        result = (
            supabase.table("chain_snapshots")
            .select("data, created_at")
            .eq("ticker", ticker)
            .eq("expiry", expiry)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            return jsonify({"snapshot": row["data"], "cached_at": row["created_at"]})
        return jsonify({"error": "No snapshot found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/snapshot/market")
@login_required
def api_snapshot_market():
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 503
    try:
        result = (
            supabase.table("market_snapshots")
            .select("data, created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            return jsonify({"snapshot": row["data"], "cached_at": row["created_at"]})
        return jsonify({"error": "No snapshot found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: Admin ───────────────────────────

@app.route("/api/admin/add_subscriber", methods=["POST"])
@admin_required
def admin_add_subscriber():
    if supabase is None:
        return jsonify({"error": "Supabase not configured"}), 503
    body = request.get_json(force=True) or {}
    invite_code = body.get("invite_code", "").strip()
    email       = body.get("email", "").strip()
    tier        = body.get("tier", "basic").strip()
    if not invite_code:
        return jsonify({"error": "invite_code required"}), 400
    try:
        result = supabase.table("subscribers").insert({
            "invite_code": invite_code,
            "email":       email or None,
            "tier":        tier,
            "active":      True,
        }).execute()
        return jsonify({"ok": True, "id": result.data[0]["id"] if result.data else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/refresh_market", methods=["POST"])
@admin_required
def admin_refresh_market():
    try:
        data = fetch_market_data_live()
        total_cv = sum(d["call_vol"] for d in data.values())
        total_pv = sum(d["put_vol"]  for d in data.values())
        total_co = sum(d["call_oi"]  for d in data.values())
        total_po = sum(d["put_oi"]   for d in data.values())
        vol_pcr  = round(total_pv / total_cv, 3) if total_cv > 0 else 0.0
        oi_pcr   = round(total_po / total_co, 3) if total_co > 0 else 0.0

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "indices":   data,
            "aggregate": {
                "call_vol":      total_cv,
                "put_vol":       total_pv,
                "call_oi":       total_co,
                "put_oi":        total_po,
                "vol_pcr":       vol_pcr,
                "oi_pcr":        oi_pcr,
                "sentiment_vol": sentiment(vol_pcr),
                "sentiment_oi":  sentiment(oi_pcr),
            },
        }
        _cache_set("market", payload)

        if supabase:
            supabase.table("market_snapshots").insert({"data": payload}).execute()

        return jsonify({"ok": True, "timestamp": payload["timestamp"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: VIX (FRED) ───────────────────────────

def fetch_fred_vix_data():
    # Re-read at call time so Flask reloader doesn't lose the module-level value
    api_key = os.environ.get("FRED_API_KEY", FRED_API_KEY)
    if not api_key:
        raise ValueError("FRED_API_KEY not set in .env")
    params = {
        "series_id":  FRED_VIX_ID,
        "api_key":    api_key,
        "file_type":  "json",
        "sort_order": "asc",
    }
    resp = requests.get(FRED_BASE, params=params, timeout=(10, 30))
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for obs in data.get("observations", []):
        val = obs.get("value", ".")
        if val == ".":
            continue
        try:
            rows.append({"date": obs["date"], "value": float(val)})
        except (ValueError, KeyError):
            continue
    if not rows:
        raise ValueError("FRED returned no VIX data")
    # Last 252 trading days
    rows = rows[-252:]
    return rows


@app.route("/api/vix")
@login_required
def api_vix():
    cached = _cache_get("vix")
    if cached:
        return jsonify(cached)
    try:
        rows = fetch_fred_vix_data()
        latest = rows[-1]["value"] if rows else None

        def regime(v):
            if v < 15:  return "COMPLACENT"
            if v < 25:  return "NORMAL"
            if v < 35:  return "ELEVATED"
            return "FEAR/PANIC"

        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "series":    "VIXCLS",
            "rows":      rows,
            "summary": {
                "latest":    latest,
                "regime":    regime(latest) if latest else None,
                "avg_30d":   round(sum(r["value"] for r in rows[-30:])  / min(30,  len(rows)), 2) if rows else None,
                "avg_90d":   round(sum(r["value"] for r in rows[-90:])  / min(90,  len(rows)), 2) if rows else None,
                "high_1y":   max(r["value"] for r in rows) if rows else None,
                "low_1y":    min(r["value"] for r in rows) if rows else None,
            },
        }
        _cache_set("vix", payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────── API: CBOE Sentiment ───────────────────────────

def fetch_cboe_sent_data(series):
    symbol, desc = CBOE_SENT_TICKERS.get(series, ("^SKEW", ""))
    import yfinance as yf
    raw = yf.download(symbol, period="1y", interval="1d",
                      progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[("Close", symbol)]
    else:
        close = raw["Close"]
    df = close.dropna().reset_index()
    df.columns = ["date", "value"]
    df["date"]  = df["date"].dt.strftime("%Y-%m-%d")
    df["value"] = df["value"].round(2)
    return df.to_dict(orient="records"), desc


@app.route("/api/cboe")
@login_required
def api_cboe():
    series = request.args.get("series", "SKEW").strip().upper()
    if series not in CBOE_SENT_TICKERS:
        return jsonify({"error": f"Unknown series: {series}. Choose from {list(CBOE_SENT_TICKERS)}"}), 400
    cache_key = f"cboe_{series}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)
    try:
        rows, desc = fetch_cboe_sent_data(series)
        symbol = CBOE_SENT_TICKERS[series][0]
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "series":    series,
            "symbol":    symbol,
            "desc":      desc,
            "rows":      rows,
        }
        _cache_set(cache_key, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
