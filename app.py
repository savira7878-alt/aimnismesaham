import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Coba import ta, fallback kalau belum terinstall
try:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD, SMAIndicator, EMAIndicator
    HAS_TA = True
except ImportError:
    HAS_TA = False

st.set_page_config(
    page_title="AimnismeSaham — AI Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== STYLE MOBILE-FRIENDLY ======================
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
    }
    div[data-testid="stDataFrame"] {
        font-size: 0.85em;
    }
    .big-score {
        font-size: 1.8em;
        font-weight: bold;
        color: #00C853;
    }
</style>
""", unsafe_allow_html=True)

# ====================== DAFTAR SAHAM ======================
# ±130 saham likuid IDX (LQ45 + IDX80 + saham aktif lainnya)
TICKERS = [
    # Banking
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BBTN.JK", "BRIS.JK", "BJTM.JK", "BJBR.JK", "PNBN.JK", "BDMN.JK",
    # Telco & Media
    "TLKM.JK", "EXCL.JK", "ISAT.JK", "TBIG.JK", "TOWR.JK", "MTEL.JK", "EMTK.JK", "MNCN.JK", "SCMA.JK",
    # Consumer & Retail
    "ICBP.JK", "INDF.JK", "UNVR.JK", "MYOR.JK", "GGRM.JK", "HMSP.JK", "KLBF.JK", "SIDO.JK", "KAEF.JK",
    "AMRT.JK", "MAPI.JK", "ACES.JK", "LPPF.JK", "RALS.JK", "ERAA.JK", "MAPA.JK", "MIDI.JK",
    # Automotive & Industrial
    "ASII.JK", "UNTR.JK", "AUTO.JK", "IMAS.JK", "GJTL.JK", "INDS.JK", "SMSM.JK",
    # Mining & Energy
    "ADRO.JK", "PTBA.JK", "ITMG.JK", "HRUM.JK", "BUMI.JK", "DOID.JK", "PTRO.JK", "DEWA.JK",
    "ANTM.JK", "INCO.JK", "MDKA.JK", "TINS.JK", "PSAB.JK", "BRMS.JK",
    "AMMN.JK", "CUAN.JK", "MBMA.JK", "NCKL.JK", "NICL.JK",
    "MEDC.JK", "ENRG.JK", "ELSA.JK", "AKRA.JK", "PGAS.JK",
    # Plantation & Agri
    "AALI.JK", "LSIP.JK", "SIMP.JK", "SSMS.JK", "PALM.JK", "DSNG.JK", "JAWA.JK", "TAPG.JK",
    "CPIN.JK", "JPFA.JK", "MAIN.JK", "WIIM.JK",
    # Property & Construction
    "BSDE.JK", "CTRA.JK", "PWON.JK", "SMRA.JK", "DMAS.JK", "APLN.JK", "ASRI.JK",
    "WIKA.JK", "WSKT.JK", "PTPP.JK", "ADHI.JK", "JKON.JK", "TOTL.JK",
    # Cement & Basic
    "SMGR.JK", "INTP.JK", "SMCB.JK", "BRPT.JK", "TPIA.JK", "FPNI.JK", "INKP.JK", "TKIM.JK",
    # Others / Active
    "GOTO.JK", "BUKA.JK", "EMTK.JK", "WIFI.JK", "JSMR.JK", "BIRD.JK", "BULL.JK",
    "ESSA.JK", "BYAN.JK", "SGER.JK", "PGEO.JK", "KEJU.JK", "SRTG.JK", "BBHI.JK",
    "HEAL.JK", "MIKA.JK", "SILO.JK", "SRAJ.JK", "SAME.JK"
]

# ====================== FUNGSI DATA ======================
@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 menit
def get_history(ticker: str, period: str = "6mo"):
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 40:
            return None
        df = df.dropna()
        return df
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("shortName") or info.get("longName") or ticker.replace(".JK", ""),
            "sector": info.get("sector") or info.get("industry") or "-",
            "pe": info.get("trailingPE") or info.get("forwardPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {
            "name": ticker.replace(".JK", ""),
            "sector": "-",
            "pe": None,
            "pb": None,
            "roe": None,
            "market_cap": None,
        }

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # RSI
    if HAS_TA:
        rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]
    else:
        rsi = calc_rsi(close).iloc[-1]

    # MACD
    if HAS_TA:
        macd_ind = MACD(close=close)
        macd_line = macd_ind.macd().iloc[-1]
        signal = macd_ind.macd_signal().iloc[-1]
        hist = macd_ind.macd_diff().iloc[-1]
    else:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = (ema12 - ema26).iloc[-1]
        signal = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]
        hist = macd_line - signal

    # Moving Averages
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]

    # Volume
    vol_sma = volume.rolling(20).mean().iloc[-1]
    vol_ratio = float(volume.iloc[-1] / vol_sma) if vol_sma and vol_sma > 0 else 1.0

    # Changes
    price = float(close.iloc[-1])
    change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0
    change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0
    change_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0

    # ===== ATR (untuk Stop Loss & Take Profit) =====
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())

    # ===== Support & Resistance sederhana (20 hari terakhir) =====
    lookback = min(20, len(high))
    recent_high = float(high.iloc[-lookback:].max())
    recent_low = float(low.iloc[-lookback:].min())
    # Pivot-style
    resistance = recent_high
    support = recent_low

    # Entry zone (dekat support jika uptrend, atau area tengah)
    if price > sma20:
        entry_low = max(support, price * 0.985)
        entry_high = price * 1.005
    else:
        entry_low = support
        entry_high = (support + resistance) / 2

    # Stop Loss & Take Profit suggestion
    stop_loss = price - (1.5 * atr)
    take_profit_1 = price + (2.0 * atr)   # RR 1:1.3
    take_profit_2 = price + (3.0 * atr)   # RR lebih besar

    # Pastikan SL tidak di bawah support terlalu jauh
    if stop_loss < support * 0.98:
        stop_loss = support * 0.98

    return {
        "price": price,
        "rsi": float(rsi) if not np.isnan(rsi) else 50.0,
        "macd": float(macd_line),
        "macd_signal": float(signal),
        "macd_hist": float(hist),
        "sma20": float(sma20),
        "sma50": float(sma50),
        "ema20": float(ema20),
        "vol_ratio": vol_ratio,
        "change_1d": change_1d,
        "change_5d": change_5d,
        "change_20d": change_20d,
        "close": float(close.iloc[-1]),
        "prev_close": float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1]),
        "atr": atr,
        "support": support,
        "resistance": resistance,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_loss": stop_loss,
        "tp1": take_profit_1,
        "tp2": take_profit_2,
    }

def ai_score(tech: dict, fund: dict) -> int:
    """
    Skor AI 0-100
    Kombinasi teknikal (utama) + fundamental
    """
    score = 50.0

    # ----- Teknikal (bobot besar) -----
    # RSI
    rsi = tech["rsi"]
    if 35 <= rsi <= 55:
        score += 9
    elif 25 <= rsi < 35:
        score += 11          # oversold menarik
    elif 55 < rsi <= 65:
        score += 5
    elif rsi < 25:
        score += 7
    elif rsi > 75:
        score -= 10
    elif rsi > 70:
        score -= 5

    # Trend kekuatan
    price = tech["price"]
    if price > tech["sma20"] > tech["sma50"]:
        score += 14          # strong uptrend
    elif price > tech["sma20"]:
        score += 7
    elif price < tech["sma20"] < tech["sma50"]:
        score -= 10          # downtrend
    else:
        score -= 3

    # MACD
    if tech["macd_hist"] > 0 and tech["macd"] > tech["macd_signal"]:
        score += 9
    elif tech["macd_hist"] > 0:
        score += 4
    else:
        score -= 6

    # Volume confirmation
    if tech["vol_ratio"] >= 1.8:
        score += 8
    elif tech["vol_ratio"] >= 1.3:
        score += 4
    elif tech["vol_ratio"] < 0.7:
        score -= 3

    # Momentum
    if tech["change_5d"] > 5:
        score += 6
    elif tech["change_5d"] > 2:
        score += 3
    elif tech["change_5d"] < -7:
        score -= 7
    elif tech["change_5d"] < -3:
        score -= 3

    # ----- Fundamental (bobot sedang) -----
    pe = fund.get("pe")
    pb = fund.get("pb")
    roe = fund.get("roe")

    if pe is not None:
        if 6 <= pe <= 18:
            score += 7
        elif 3 <= pe < 6:
            score += 4
        elif pe > 35:
            score -= 6
        elif pe > 25:
            score -= 3

    if pb is not None:
        if 0.6 <= pb <= 2.2:
            score += 5
        elif pb < 0.7:
            score += 3
        elif pb > 4:
            score -= 4

    if roe is not None:
        if roe >= 0.18:
            score += 5
        elif roe >= 0.12:
            score += 3
        elif roe < 0.05:
            score -= 3

    return int(max(0, min(100, round(score))))


def smart_money_score(tech: dict) -> int:
    """
    Smart Money Score (Proxy Bandarmologi) 0-100
    Berdasarkan volume anomaly + price action
    """
    score = 40.0
    vol = tech["vol_ratio"]
    change = tech["change_1d"]
    change5 = tech["change_5d"]

    # 1. Relative Volume (bobot besar)
    if vol >= 3.0:
        score += 25
    elif vol >= 2.0:
        score += 18
    elif vol >= 1.5:
        score += 12
    elif vol >= 1.2:
        score += 6
    elif vol < 0.6:
        score -= 10

    # 2. Volume + Price Action (Akumulasi vs Distribusi)
    if vol >= 1.5 and change > 1.5:
        score += 18          # Volume tinggi + harga naik kuat → akumulasi
    elif vol >= 1.5 and change > 0.3:
        score += 10
    elif vol >= 1.5 and change < -1.5:
        score -= 15          # Volume tinggi + harga turun → distribusi
    elif vol >= 1.5 and change < -0.5:
        score -= 8

    # 3. Momentum 5 hari mendukung
    if change5 > 4 and vol >= 1.3:
        score += 10
    elif change5 < -5 and vol >= 1.3:
        score -= 8

    # 4. Harga di atas MA (konfirmasi trend)
    if tech["price"] > tech["sma20"]:
        score += 7
    else:
        score -= 5

    return int(max(0, min(100, round(score))))


def score_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    elif score >= 70:
        return "🟡"
    elif score >= 60:
        return "🟠"
    else:
        return "⚪"


def get_recommendation(score: int, smart: int, tech: dict, ihsg_status: str) -> dict:
    """Tentukan Recommended Action mirip STOCKS.LY"""
    price = tech["price"]
    sma20 = tech["sma20"]
    sma50 = tech["sma50"]
    rsi = tech["rsi"]
    vol = tech["vol_ratio"]
    atr_pct = (tech["atr"] / price * 100) if price > 0 else 0

    # Kondisi harga
    price_above_ma20 = price > sma20
    ma20_above_ma50 = sma20 > sma50
    vol_ok = vol >= 0.8
    atr_ok = atr_pct < 8  # volatilitas tidak ekstrem

    conditions_ok = sum([price_above_ma20, ma20_above_ma50, vol_ok, atr_ok])

    # Decision logic
    if score >= 82 and smart >= 70 and price_above_ma20 and ihsg_status != "Bearish":
        action = "BUY"
        color = "green"
        reason = "Skor AI & Smart Money kuat, tren naik, kondisi harga mendukung."
    elif score >= 70 and smart >= 55 and price_above_ma20:
        action = "WATCHLIST"
        color = "orange"
        reason = "Menarik, tapi tunggu konfirmasi volume atau pullback ke Entry Zone."
    elif score >= 60 or smart >= 50:
        action = "WAIT"
        color = "gray"
        reason = "Timing belum optimal atau sinyal masih campuran. Pantau dulu."
    else:
        action = "AVOID"
        color = "red"
        reason = "Skor rendah / tren lemah. Hindari untuk sementara."

    # Risk level 1-5
    risk = 3
    if score >= 80 and smart >= 70 and price_above_ma20:
        risk = 2  # lebih terkendali
    if not price_above_ma20 or atr_pct > 7 or rsi > 75:
        risk = 4
    if score < 55 or (ihsg_status == "Bearish" and not price_above_ma20):
        risk = 5

    risk_label = {1: "SANGAT TERKENDALI", 2: "TERKENDALI", 3: "NETRAL", 4: "BERISIKO", 5: "SANGAT BERISIKO"}

    return {
        "action": action,
        "color": color,
        "reason": reason,
        "risk": risk,
        "risk_label": risk_label.get(risk, "NETRAL"),
        "price_above_ma20": price_above_ma20,
        "ma20_above_ma50": ma20_above_ma50,
        "vol_ok": vol_ok,
        "atr_ok": atr_ok,
        "atr_pct": atr_pct,
        "conditions_ok": conditions_ok,
    }


@st.cache_data(ttl=1800, show_spinner=False)
def get_ihsg_trend():
    """Cek tren IHSG sederhana"""
    try:
        df = yf.download("^JKSE", period="3mo", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < 30:
            return {"status": "Unknown", "change_5d": 0, "above_ma": True}
        close = df["Close"].astype(float)
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
        price = close.iloc[-1]
        change_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0

        if price > sma20 > sma50 and change_5d > 0:
            status = "Bullish"
        elif price < sma20 < sma50 and change_5d < 0:
            status = "Bearish"
        else:
            status = "Sideways"

        return {
            "status": status,
            "change_5d": change_5d,
            "above_ma": price > sma20,
            "price": float(price)
        }
    except Exception:
        return {"status": "Unknown", "change_5d": 0, "above_ma": True, "price": 0}

# ====================== UI ======================
st.title("📈 AimnismeSaham")
st.caption(f"AI Stock Screener IDX  •  {datetime.now().strftime('%d %b %Y • %H:%M')} WIB  •  Yahoo Finance  •  Bukan saran investasi")

# Sidebar filter
with st.sidebar:
    st.header("⚙️ Filter")
    min_score = st.slider("Skor AI minimum", 0, 100, 60, 1)
    min_smart = st.slider("Smart Money Score minimum", 0, 100, 50, 1)
    only_uptrend = st.checkbox("Hanya yang di atas SMA20", value=True)
    respect_ihsg = st.checkbox("Hormati tren IHSG (hindari jika Bearish)", value=True)
    max_show = st.slider("Jumlah saham ditampilkan", 5, 40, 12)
    st.markdown("---")
    st.markdown("**Keterangan Skor**")
    st.markdown("""
    **Skor AI** : Teknikal + Fundamental  
    **Smart $** : Proxy volume / smart money  
    **SL / TP** : Berdasarkan ATR  
    """)
    st.markdown("---")
    st.caption("Support • Resistance • Entry Zone • SL/TP")

# ====================== TOMBOL SCAN YANG JELAS ======================
st.markdown("---")
st.markdown("### 🔍 Cari Saham Potensial Hari Ini")
st.markdown("Tekan tombol di bawah untuk memulai screening saham berdasarkan **Teknikal + Fundamental + Skor AI**")

scan = st.button(
    "🚀  SCAN SAHAM SEKARANG",
    type="primary",
    use_container_width=True,
    help="Klik untuk memulai analisa seluruh saham"
)

st.caption("Quick Scan ±40 detik | Full Scan 2–4 menit. Jangan tutup halaman saat proses berjalan.")

# Pilihan mode scan
scan_mode = st.radio(
    "Mode Scan:",
    ["⚡ Quick Scan (40 saham paling likuid)", "🔍 Full Scan (semua ±120 saham)"],
    horizontal=True,
    index=0
)

if scan:
    # Cek tren IHSG dulu
    ihsg = get_ihsg_trend()
    
    if respect_ihsg and ihsg["status"] == "Bearish":
        st.warning(f"⚠️ **IHSG sedang Bearish** (5D: {ihsg['change_5d']:+.1f}%). Rekomendasi dibatasi.")
    
    # Pilih daftar ticker sesuai mode
    if "Quick" in scan_mode:
        scan_list = TICKERS[:42]   # 42 saham paling likuid di awal list
    else:
        scan_list = TICKERS

    results = []
    progress = st.progress(0)
    status = st.empty()

    total = len(scan_list)
    for i, ticker in enumerate(scan_list):
        kode = ticker.replace(".JK", "")
        status.markdown(f"**Scanning** `{kode}`  ({i+1}/{total})")
        
        df = get_history(ticker)
        if df is None:
            progress.progress((i + 1) / total)
            continue

        tech = calculate_indicators(df)
        smart = smart_money_score(tech)

        # Filter teknis dulu (cepat) — baru ambil fundamental kalau lolos
        if only_uptrend and tech["price"] < tech["sma20"]:
            progress.progress((i + 1) / total)
            continue
        if smart < min_smart:
            progress.progress((i + 1) / total)
            continue

        # Ambil fundamental hanya untuk kandidat yang sudah lolos filter volume/trend
        fund = get_info(ticker)
        score = ai_score(tech, fund)

        if score < min_score:
            progress.progress((i + 1) / total)
            continue
        if respect_ihsg and ihsg["status"] == "Bearish" and score < 75:
            progress.progress((i + 1) / total)
            continue

        rec = get_recommendation(score, smart, tech, ihsg["status"])

        results.append({
            "Kode": kode,
            "Nama": fund["name"][:28],
            "Harga": tech["price"],
            "1D%": tech["change_1d"],
            "5D%": tech["change_5d"],
            "RSI": tech["rsi"],
            "Vol": tech["vol_ratio"],
            "Skor AI": score,
            "Smart $": smart,
            "Support": tech["support"],
            "Resistance": tech["resistance"],
            "Entry": f"{tech['entry_low']:.0f}-{tech['entry_high']:.0f}",
            "SL": tech["stop_loss"],
            "TP1": tech["tp1"],
            "TP2": tech["tp2"],
            "PE": fund.get("pe"),
            "PBV": fund.get("pb"),
            "ROE": fund.get("roe"),
            "Sektor": fund.get("sector") or "-",
            "Action": rec["action"],
            "Reason": rec["reason"],
            "Risk": rec["risk"],
            "RiskLabel": rec["risk_label"],
            "ATRpct": rec["atr_pct"],
            "Price>MA20": rec["price_above_ma20"],
            "MA20>MA50": rec["ma20_above_ma50"],
            "VolOK": rec["vol_ok"],
            "ATROK": rec["atr_ok"],
            "CondOK": rec["conditions_ok"],
            "SMA20": tech["sma20"],
            "SMA50": tech["sma50"],
        })
        progress.progress((i + 1) / total)

    status.empty()
    progress.empty()

    if not results:
        st.warning("Tidak ada saham yang lolos filter. Coba turunkan skor minimum atau matikan filter uptrend.")
    else:
        df = pd.DataFrame(results)
        df = df.sort_values("Skor AI", ascending=False).head(max_show).reset_index(drop=True)

        # Tampilkan top 3 highlight
        # Status IHSG
        ihsg_icon = "🟢" if ihsg["status"] == "Bullish" else ("🔴" if ihsg["status"] == "Bearish" else "🟡")
        st.info(f"{ihsg_icon} **IHSG**: {ihsg['status']} | 5D: {ihsg['change_5d']:+.1f}%")

        st.success(f"✅ Scan selesai! Ditemukan **{len(df)} saham** potensial.")
        st.markdown("### 🏆 Rekomendasi Saham Potensial")
        st.caption("Urut berdasarkan Skor AI • Dilengkapi Keputusan Strategi, Entry Zone, SL & TP")

        # ===== TOP 3 CARDS =====
        top_cols = st.columns(min(3, len(df)))
        for idx, col in enumerate(top_cols):
            if idx < len(df):
                row = df.iloc[idx]
                action = row["Action"]
                action_emoji = {"BUY": "🟢 BUY", "WATCHLIST": "🟡 WATCHLIST", "WAIT": "⚪ WAIT", "AVOID": "🔴 AVOID"}.get(action, action)
                with col:
                    st.markdown(f"### {score_color(row['Skor AI'])} {row['Kode']}")
                    st.metric("Harga", f"{row['Harga']:,.0f}", delta=f"{row['5D%']:+.1f}% (5D)")
                    st.markdown(f"**{action_emoji}**")
                    st.caption(f"AI {row['Skor AI']} | Smart {row['Smart $']} | Risk: {row['RiskLabel']}")
                    st.caption(f"Entry: {row['Entry']}")
                    st.caption(f"SL: {row['SL']:,.0f} → TP1: {row['TP1']:,.0f}")

        st.markdown("---")

        # ===== DETAIL ANALISIS (mirip STOCKS.LY) =====
        st.markdown("### 📊 Detail Analisis per Saham")
        st.caption("Klik expander untuk melihat Ringkasan Strategi, Kondisi Harga, Entry Area & Keputusan")

        for idx, row in df.iterrows():
            action = row["Action"]
            action_color = {"BUY": "green", "WATCHLIST": "orange", "WAIT": "gray", "AVOID": "red"}.get(action, "gray")
            with st.expander(f"{score_color(row['Skor AI'])} **{row['Kode']}** — {row['Nama']}  |  {action}", expanded=(idx == 0)):
                
                # Header
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.markdown(f"**{row['Nama']}**")
                    st.caption(f"Sektor: {row['Sektor']}")
                with c2:
                    st.metric("Skor AI", f"{row['Skor AI']}")
                with c3:
                    st.metric("Smart Money", f"{row['Smart $']}")

                st.markdown("---")

                # Ringkasan Strategi + Kondisi Harga
                colA, colB = st.columns(2)

                with colA:
                    st.markdown("#### ⚡ Ringkasan Strategi")
                    pe_txt = f"{row['PE']:.1f}x" if pd.notna(row['PE']) else "–"
                    roe_txt = f"{row['ROE']*100:.1f}%" if pd.notna(row['ROE']) and row['ROE'] is not None else "–"
                    st.write(f"**Fundamental:** PE {pe_txt} | ROE {roe_txt}")
                    st.write(f"**Smart Money Zone:** {'Kuat' if row['Smart $'] >= 70 else ('Sedang' if row['Smart $'] >= 50 else 'Lemah')}")
                    st.write(f"**Signal Reliability:** {'High' if row['Skor AI'] >= 80 and row['Smart $'] >= 65 else 'Medium' if row['Skor AI'] >= 65 else 'Low'}")
                    
                    # Risk bar sederhana
                    risk = int(row["Risk"])
                    risk_colors = ["🟢", "🟢", "🟡", "🟠", "🔴"]
                    st.write(f"**Risiko:** {risk_colors[risk-1]} {row['RiskLabel']} ({risk}/5)")

                with colB:
                    st.markdown("#### 📈 Kondisi Harga Saat Ini")
                    st.write(f"**Last Price:** {row['Harga']:,.0f}")
                    st.write(f"**Volatility (ATR):** {row['ATRpct']:.1f}%")
                    st.write(f"**MA20:** {row['SMA20']:,.0f} | **MA50:** {row['SMA50']:,.0f}")
                    
                    badges = []
                    badges.append("✅ Price > MA20" if row["Price>MA20"] else "❌ Price < MA20")
                    badges.append("✅ MA20 > MA50" if row["MA20>MA50"] else "❌ MA20 < MA50")
                    badges.append("✅ Volatility OK" if row["ATROK"] else "⚠️ Volatility Tinggi")
                    badges.append("✅ Liquidity OK" if row["VolOK"] else "⚠️ Volume Sepi")
                    st.write(" • ".join(badges))
                    
                    suitable = "SUITABLE" if row["CondOK"] >= 3 and row["Action"] in ["BUY", "WATCHLIST"] else "NOT SUITABLE"
                    st.markdown(f"**Status:** `{suitable}`")

                st.markdown("---")

                # Entry Area + Keputusan
                colC, colD = st.columns(2)
                with colC:
                    st.markdown("#### 🎯 Entry Area")
                    if action == "AVOID":
                        st.error("HINDARI EMITEN INI untuk sementara")
                        st.caption("Skor atau kondisi belum mendukung entry.")
                    elif action == "WAIT":
                        st.warning("Belum ada sinyal entry yang kuat")
                        st.caption(f"Pantau zona {row['Entry']}")
                    else:
                        st.success(f"Entry Zone: **{row['Entry']}**")
                        st.caption(f"Support: {row['Support']:,.0f} | Resistance: {row['Resistance']:,.0f}")
                        st.write(f"SL: **{row['SL']:,.0f}** | TP1: **{row['TP1']:,.0f}** | TP2: **{row['TP2']:,.0f}**")

                with colD:
                    st.markdown("#### 📋 Keputusan Strategi")
                    if action == "BUY":
                        st.success(f"**RECOMMENDED ACTION: BUY**")
                    elif action == "WATCHLIST":
                        st.warning(f"**RECOMMENDED ACTION: WATCHLIST**")
                    elif action == "WAIT":
                        st.info(f"**RECOMMENDED ACTION: WAIT**")
                    else:
                        st.error(f"**RECOMMENDED ACTION: AVOID**")
                    st.write(row["Reason"])

                st.caption("⚠️ Data Bandar Flow & Dana Asing asli tidak tersedia secara gratis. Smart Money Score adalah proxy berbasis volume.")

        st.markdown("---")
        st.markdown(f"### 📋 Tabel Ringkas ({len(df)} saham)")

        # Tabel ringkas
        show_cols = ["Kode", "Nama", "Harga", "5D%", "Skor AI", "Smart $", "Action", "Entry", "SL", "TP1"]
        display = df[show_cols].copy()
        display["Harga"] = display["Harga"].map(lambda x: f"{x:,.0f}")
        display["5D%"] = display["5D%"].map(lambda x: f"{x:+.1f}%")
        display["SL"] = display["SL"].map(lambda x: f"{x:,.0f}")
        display["TP1"] = display["TP1"].map(lambda x: f"{x:,.0f}")

        st.dataframe(display, use_container_width=True, hide_index=True)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download hasil lengkap (CSV)",
            csv,
            file_name=f"aimnismesaham_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 Tekan tombol **🚀 SCAN SAHAM SEKARANG** di atas untuk memulai.")
    st.markdown("""
    ### Fitur AimnismeSaham (terinspirasi STOCKS.LY)
    1. **Skor AI** + **Smart Money Score**
    2. **Recommended Action** → BUY / WATCHLIST / WAIT / AVOID
    3. **Ringkasan Strategi** (Fundamental + Risk Level)
    4. **Kondisi Harga** (Price vs MA, Volatility, Liquidity)
    5. **Entry Area** + Support / Resistance + SL & TP
    6. **Filter tren IHSG**

    **Catatan:**  
    Data Bandar Flow & Dana Asing asli **tidak tersedia gratis**.  
    Smart Money Score adalah **proxy** berbasis volume & price action.
    """)

st.markdown("---")
st.caption("⚠️ Tool edukasi pribadi. Bukan rekomendasi investasi. Data bisa delay. Selalu lakukan riset mandiri.")
