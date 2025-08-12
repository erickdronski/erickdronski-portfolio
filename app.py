import os
import json
import threading
import sqlite3
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
import yfinance as yf
import plotly.express as px

# ===== Optional deps =====
# FastAPI for webhooks
try:
    from fastapi import FastAPI, Request, Header, HTTPException
    import uvicorn
    FASTAPI_AVAILABLE = True
except Exception:
    FASTAPI_AVAILABLE = False

# VADER for sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except Exception:
    VADER_AVAILABLE = False

# Supabase for cloud persistence
try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Tech News + Markets + Alerts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Globals / Secrets
# =========================
DB_PATH = os.getenv("ALERTS_DB", "alerts.db")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", "8000"))
WEBHOOK_TOKEN = os.getenv("WEBHOOK_TOKEN", st.secrets.get("webhook_token", None) if hasattr(st, "secrets") else None)

# Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SB = None
if SUPABASE_AVAILABLE and SUPABASE_URL and (SUPABASE_ANON_KEY or SUPABASE_SERVICE_KEY):
    try:
        SB = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY)
    except Exception:
        SB = None

# Telegram push
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ON = bool(os.getenv("TELEGRAM_ENABLED", "1").lower() not in ("0", "false", "no")) and bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

# Friendly names for tickers
FRIENDLY = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^NDX": "Nasdaq 100",
    "NQ=F": "NAS100 Mini (E-mini)",
}

# =========================
# SQLite helpers (fallback if no Supabase)
# =========================
def init_db(db_path: str = DB_PATH):
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at TEXT,
            ticker TEXT,
            price REAL,
            signal TEXT,
            strategy TEXT,
            message TEXT,
            raw_json TEXT
        )
        """
    )
    con.commit()
    return con

DB = init_db()

# =========================
# Utilities
# =========================
def telegram_notify(text: str) -> bool:
    if not (TELEGRAM_ON and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
            timeout=10,
        )
        return r.ok
    except Exception:
        return False

# Insert alert -> Supabase if available, else SQLite (also optionally Telegram)
def insert_alert(payload: dict):
    ticker = str(payload.get("ticker") or payload.get("symbol") or "?")
    price = payload.get("price") or payload.get("close") or payload.get("last")
    signal = payload.get("signal") or payload.get("action")
    strategy = payload.get("strategy") or payload.get("strategy_name") or payload.get("comment")
    message = payload.get("message") or payload.get("note")
    now_utc = datetime.utcnow().isoformat()

    # Dual write: Supabase first, fallback to SQLite
    if SB is not None:
        try:
            SB.table("alerts").insert({
                "received_at": now_utc,
                "ticker": ticker,
                "price": float(price) if price is not None else None,
                "signal": str(signal) if signal is not None else None,
                "strategy": str(strategy) if strategy is not None else None,
                "message": str(message) if message is not None else None,
                "raw_json": json.dumps(payload),
            }).execute()
        except Exception:
            pass

    DB.execute(
        "INSERT INTO alerts (received_at, ticker, price, signal, strategy, message, raw_json) VALUES (?,?,?,?,?,?,?)",
        (now_utc, ticker, float(price) if price is not None else None, str(signal) if signal is not None else None, str(strategy) if strategy is not None else None, str(message) if message is not None else None, json.dumps(payload)),
    )
    DB.commit()

    # Telegram
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    telegram_notify(
        f"📨 TV Alert
{ticker} {signal or ''} @ {price}
Strategy: {strategy or '-'}
{message or ''}
{local_time}"
    )

# Fetch alerts (Supabase preferred)
def fetch_alerts(limit: int = 200) -> pd.DataFrame:
    df = pd.DataFrame()
    if SB is not None:
        try:
            res = SB.table("alerts").select("id, received_at, ticker, price, signal, strategy, message").order("id", desc=True).limit(limit).execute()
            if hasattr(res, "data") and res.data:
                df = pd.DataFrame(res.data)
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        df = pd.read_sql_query(
            "SELECT id, received_at, ticker, price, signal, strategy, message FROM alerts ORDER BY id DESC LIMIT ?",
            DB,
            params=(limit,),
        )
    if not df.empty:
        try:
            df["received_at"] = pd.to_datetime(df["received_at"]).dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
        except Exception:
            pass
    return df

# =========================
# FastAPI webhook app (exported for Render)
# =========================
fastapi_app = None
if FASTAPI_AVAILABLE:
    fastapi_app = FastAPI()

    @fastapi_app.post("/webhook/tradingview")
    async def tradingview_webhook(request: Request, authorization: str | None = Header(default=None)):
        token_ok = False
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        if WEBHOOK_TOKEN:
            if authorization and authorization.strip() == f"Bearer {WEBHOOK_TOKEN}":
                token_ok = True
            if request.query_params.get("token") == WEBHOOK_TOKEN:
                token_ok = True
        else:
            token_ok = True  # dev open mode

        if not token_ok:
            raise HTTPException(status_code=401, detail="Invalid or missing token")

        try:
            insert_alert(payload)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")
        return {"ok": True}

# Start webhook server in a background thread (for local dev only)
def start_webhook_server():
    if not FASTAPI_AVAILABLE:
        st.warning("FastAPI/uvicorn not installed. Add fastapi and uvicorn to requirements to enable webhooks.")
        return
    if "_webhook_running" in st.session_state and st.session_state["_webhook_running"]:
        return

    def _run():
        uvicorn.run(fastapi_app, host="0.0.0.0", port=WEBHOOK_PORT, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    st.session_state["_webhook_running"] = True

# =========================
# Scrapers
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def scrape_hn_frontpage(max_items: int = 30) -> pd.DataFrame:
    url = "https://news.ycombinator.com/"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for a in soup.select("span.titleline > a")[:max_items]:
        title = a.get_text(strip=True)
        link = a.get("href")
        domain = urlparse(link).netloc or "news.ycombinator.com"
        rows.append({"title": title, "url": link, "source": domain})
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)
def scrape_theverge(max_items: int = 20) -> pd.DataFrame:
    url = "https://www.theverge.com/tech"
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    for article in soup.select("div.duet--content-cards--content-card a[data-analytics-link]"):
        title = article.get_text(strip=True)
        link = article.get("href")
        if not title or not link:
            continue
        rows.append({
            "title": title,
            "url": link if link.startswith("http") else f"https://www.theverge.com{link}",
            "source": "www.theverge.com",
        })
        if len(rows) >= max_items:
            break
    return pd.DataFrame(rows)

# =========================
# Market Data
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def get_prices(tickers: list[str], period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"].copy()
    else:
        data = data[["Close"]].rename(columns={"Close": tickers[0]})
    data.index = pd.to_datetime(data.index)
    tidy = data.reset_index().melt(id_vars=["Date"], var_name="Ticker", value_name="Close")
    # Add friendly label
    tidy["Label"] = tidy["Ticker"].map(FRIENDLY).fillna(tidy["Ticker"])
    return tidy

@st.cache_data(ttl=300, show_spinner=False)
def get_live_quotes(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
        except Exception:
            info = None
        if not info:
            continue
        rows.append({
            "Ticker": t,
            "Name": FRIENDLY.get(t, t),
            "Last": info.get("last_price"),
            "Open": info.get("open"),
            "High": info.get("day_high"),
            "Low": info.get("day_low"),
            "Prev Close": info.get("previous_close"),
            "Currency": info.get("currency"),
        })
    return pd.DataFrame(rows)

# =========================
# Sidebar Controls
# =========================
st.sidebar.title("Controls")
news_sources = st.sidebar.multiselect(
    "News sources",
    options=["Hacker News", "The Verge"],
    default=["Hacker News", "The Verge"],
)

# Sentiment toggle
sentiment_on = st.sidebar.checkbox("Analyze news sentiment (VADER)", value=True)

# Telegram toggle
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    new_val = st.sidebar.checkbox("Telegram push on new alerts", value=TELEGRAM_ON)
    TELEGRAM_ON = bool(new_val)
else:
    st.sidebar.caption("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to enable Telegram pushes.")

# Webhook server control (local dev only)
run_webhook_local = st.sidebar.checkbox("Run local webhook server (dev)")
if run_webhook_local:
    start_webhook_server()
    st.sidebar.success(f"Webhook listening on :{WEBHOOK_PORT} /webhook/tradingview")
    if WEBHOOK_TOKEN:
        st.sidebar.write("Auth: Bearer token or ?token=… enabled")
    else:
        st.sidebar.warning("No WEBHOOK_TOKEN set — endpoint is open (dev only)")

# Tickers (now includes top 3 indices + NAS100 mini futures)
default_tickers = ["NVDA", "AAPL", "MSFT", "META", "TSLA", "^NDX", "^GSPC", "^DJI", "NQ=F"]
user_tickers = st.sidebar.text_input(
    "Tickers (comma-separated)", value=", ".join(default_tickers)
)
period = st.sidebar.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
interval = st.sidebar.selectbox("Candle interval", ["1d", "1h", "30m", "15m"], index=0)

col_refresh1, col_refresh2 = st.sidebar.columns(2)
refresh_news = col_refresh1.button("Refresh News")
refresh_prices = col_refresh2.button("Refresh Prices")

st.title("📊 Tech News + Markets + Alerts")
st.caption("BeautifulSoup scraping • yfinance prices • TradingView webhook inbox (FastAPI) • Supabase persistence • Telegram pushes • VADER sentiment")

# =========================
# News Section (with optional sentiment)
# =========================
st.subheader("📰 Tech News")
news_tabs = st.tabs(["Combined", "Hacker News", "The Verge", "Sentiment (if on)"])

# Gather feeds
try:
    hn = scrape_hn_frontpage() if ("Hacker News" in news_sources) else pd.DataFrame()
    vg = scrape_theverge() if ("The Verge" in news_sources) else pd.DataFrame()
    frames = []
    if not hn.empty:
        frames.append(hn.assign(feed="Hacker News"))
    if not vg.empty:
        frames.append(vg.assign(feed="The Verge"))
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["title", "url", "source", "feed"])    
except Exception as e:
    combined = pd.DataFrame()
    st.warning(f"News scrape failed: {e}")

with news_tabs[0]:
    st.dataframe(combined[[c for c in ["feed", "source", "title", "url"] if c in combined.columns]], use_container_width=True, hide_index=True)

with news_tabs[1]:
    if not hn.empty:
        st.dataframe(hn, use_container_width=True, hide_index=True)
    else:
        st.info("Hacker News not selected.")

with news_tabs[2]:
    if not vg.empty:
        st.dataframe(vg, use_container_width=True, hide_index=True)
    else:
        st.info("The Verge not selected.")

with news_tabs[3]:
    if sentiment_on:
        if not VADER_AVAILABLE:
            st.warning("Install vaderSentiment to enable sentiment scoring.")
        else:
            analyzer = SentimentIntensityAnalyzer()
            df = combined.copy()
            if not df.empty:
                df["sentiment"] = df["title"].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"]).astype(float)
                df["label"] = pd.cut(
                    df["sentiment"],
                    bins=[-1.0, -0.05, 0.05, 1.0],
                    labels=["Negative", "Neutral", "Positive"],
                    include_lowest=True,
                )
                st.dataframe(df[["feed", "source", "label", "sentiment", "title", "url"]], use_container_width=True, hide_index=True)
                try:
                    fig = px.histogram(df, x="sentiment", nbins=20, title="Headline Sentiment Distribution")
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
            else:
                st.info("No articles loaded yet.")
    else:
        st.info("Sentiment disabled in sidebar.")

st.markdown("---")

# =========================
# Markets Section
# =========================
st.subheader("📈 Markets")

try:
    tickers = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]
    quote_df = get_live_quotes(tickers)
    if not quote_df.empty:
        st.dataframe(quote_df, use_container_width=True, hide_index=True)
    else:
        st.info("Enter valid tickers in the sidebar (e.g., NVDA, AAPL, MSFT, ^NDX, ^GSPC, ^DJI, NQ=F).")
except Exception as e:
    st.warning(f"Quote fetch failed: {e}")

# Historical chart
try:
    hist = get_prices(tickers, period=period, interval=interval)
    if not hist.empty:
        fig = px.line(hist, x="Date", y="Close", color="Label", title=f"Historical Close — period={period}, interval={interval}")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data. Check tickers or change period/interval.")
except Exception as e:
    st.warning(f"Historical fetch failed: {e}")

st.markdown("---")

# =========================
# TradingView Alerts Inbox
# =========================
st.subheader("📨 TradingView Alerts Inbox")

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Insert test alert"):
        insert_alert({
            "ticker": "NAS100",
            "price": 19000.12,
            "signal": "BUY",
            "strategy": "Breakout 15m",
            "message": "Test alert from UI",
        })
        st.success("Inserted test alert")
    st.caption("POST JSON to /webhook/tradingview to populate this table. Include Authorization: Bearer <token> or ?token=… if set.")

with col1:
    alerts_df = fetch_alerts(limit=300)
    if alerts_df.empty:
        st.info("No alerts yet. Send a webhook or use 'Insert test alert'.")
    else:
        st.dataframe(alerts_df, use_container_width=True, hide_index=True)

st.caption("Built with Streamlit • Scraping via BeautifulSoup • Prices via yfinance • Webhooks via FastAPI • Supabase • Telegram • VADER")
