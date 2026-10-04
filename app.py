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
# LQ45 + beberapa saham likuid tambahan
TICKERS = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BBTN.JK",
    "TLKM.JK", "ASII.JK", "UNTR.JK", "ANTM.JK", "ADRO.JK",
    "PTBA.JK", "ICBP.JK", "INDF.JK", "KLBF.JK", "CPIN.JK",
    "AMRT.JK", "GOTO.JK", "AMMN.JK", "MDKA.JK", "BRPT.JK",
    "EXCL.JK", "PGAS.JK", "SMGR.JK", "INTP.JK", "JSMR.JK",
    "TBIG.JK", "ACES.JK", "MAPI.JK", "HRUM.JK", "ITMG.JK",
    "MEDC.JK", "ESSA.JK", "INKP.JK", "TKIM.JK", "JPFA.JK",
    "SIDO.JK", "UNVR.JK", "GGRM.JK", "HMSP.JK", "BRIS.JK",
    "BUMI.JK", "INCO.JK", "TPIA.JK", "BYAN.JK", "CUAN.JK"
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

def score_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    elif score >= 70:
        return "🟡"
    elif score >= 60:
        return "🟠"
    else:
        return "⚪"

# ====================== UI ======================
st.title("📈 AimnismeSaham")
st.caption(f"AI Stock Screener IDX  •  Update: {datetime.now().strftime('%d %b %Y • %H:%M')} WIB  •  Data: Yahoo Finance  •  Bukan saran investasi")

# Sidebar filter
with st.sidebar:
    st.header("⚙️ Filter")
    min_score = st.slider("Skor AI minimum", 0, 100, 62, 1)
    only_uptrend = st.checkbox("Hanya yang di atas SMA20", value=True)
    max_show = st.slider("Jumlah saham ditampilkan", 5, 40, 15)
    st.markdown("---")
    st.markdown("**Keterangan Skor AI**")
    st.markdown("""
    - **80–100** → Sangat menarik  
    - **70–79** → Menarik  
    - **60–69** → Cukup  
    - **< 60** → Lemah  
    """)
    st.markdown("---")
    st.caption("Teknikal: RSI, MACD, MA, Volume, Momentum\nFundamental: PE, PBV, ROE")

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

st.caption("Proses scan membutuhkan 30–60 detik. Mohon tunggu sampai selesai.")

if scan:
    results = []
    progress = st.progress(0)
    status = st.empty()

    total = len(TICKERS)
    for i, ticker in enumerate(TICKERS):
        kode = ticker.replace(".JK", "")
        status.markdown(f"**Scanning** `{kode}`  ({i+1}/{total})")
        
        df = get_history(ticker)
        if df is None:
            progress.progress((i + 1) / total)
            continue

        tech = calculate_indicators(df)
        fund = get_info(ticker)
        score = ai_score(tech, fund)

        # Filter
        if score < min_score:
            progress.progress((i + 1) / total)
            continue
        if only_uptrend and tech["price"] < tech["sma20"]:
            progress.progress((i + 1) / total)
            continue

        results.append({
            "Kode": kode,
            "Nama": fund["name"][:28],
            "Harga": tech["price"],
            "1D%": tech["change_1d"],
            "5D%": tech["change_5d"],
            "RSI": tech["rsi"],
            "Vol": tech["vol_ratio"],
            "PE": fund["pe"],
            "PBV": fund["pb"],
            "Skor": score,
            "Sektor": (fund["sector"] or "-")[:18],
        })
        progress.progress((i + 1) / total)
        time.sleep(0.15)  # sedikit delay biar tidak terlalu agresif ke Yahoo

    status.empty()
    progress.empty()

    if not results:
        st.warning("Tidak ada saham yang lolos filter. Coba turunkan skor minimum atau matikan filter uptrend.")
    else:
        df = pd.DataFrame(results)
        df = df.sort_values("Skor", ascending=False).head(max_show).reset_index(drop=True)

        # Tampilkan top 3 highlight
        st.success(f"✅ Scan selesai! Ditemukan **{len(df)} saham** potensial.")
        st.markdown("### 🏆 Rekomendasi Saham Potensial (Skor AI Tertinggi)")
        st.caption("Semakin tinggi skor = semakin menarik secara teknikal & fundamental saat ini")
        
        top_cols = st.columns(min(3, len(df)))
        for idx, col in enumerate(top_cols):
            if idx < len(df):
                row = df.iloc[idx]
                with col:
                    st.metric(
                        label=f"{score_color(row['Skor'])} {row['Kode']}",
                        value=f"{row['Harga']:,.0f}",
                        delta=f"{row['5D%']:+.1f}% (5D) | Skor {row['Skor']}"
                    )

        st.markdown("---")
        st.markdown(f"### 📋 Daftar Lengkap Hasil Screening ({len(df)} saham)")

        # Format tabel
        display = df.copy()
        display["Harga"] = display["Harga"].map(lambda x: f"{x:,.0f}")
        display["1D%"] = display["1D%"].map(lambda x: f"{x:+.2f}")
        display["5D%"] = display["5D%"].map(lambda x: f"{x:+.2f}")
        display["RSI"] = display["RSI"].map(lambda x: f"{x:.1f}")
        display["Vol"] = display["Vol"].map(lambda x: f"{x:.2f}x")
        display["PE"] = display["PE"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "–")
        display["PBV"] = display["PBV"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "–")
        display["Skor"] = display["Skor"].map(lambda x: f"{score_color(x)} {x}")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kode": st.column_config.TextColumn("Kode", width="small"),
                "Nama": st.column_config.TextColumn("Nama", width="medium"),
                "Skor": st.column_config.TextColumn("Skor AI", width="small"),
            }
        )

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download hasil (CSV)",
            csv,
            file_name=f"aimnismesaham_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.info("👆 Tekan tombol **🚀 SCAN SAHAM SEKARANG** di atas untuk memulai.")
    st.markdown("""
    ### Cara kerja AimnismeSaham
    1. Mengambil data harga 6 bulan terakhir dari Yahoo Finance  
    2. Menghitung indikator teknikal (RSI, MACD, SMA, Volume)  
    3. Mengambil data fundamental (PE, PBV, ROE)  
    4. Memberikan **Skor AI 0–100**  
    5. Menampilkan saham dengan skor tertinggi sebagai **Rekomendasi**

    **Tips:**
    - Jalankan setelah jam **16:00 WIB** (setelah market tutup)
    - Skor **80+** = menarik | **90+** = sangat menarik
    - Ini tool bantu analisa, **bukan** jaminan naik
    """)

st.markdown("---")
st.caption("⚠️ Tool edukasi pribadi. Bukan rekomendasi investasi. Data bisa delay. Selalu lakukan riset mandiri.")
