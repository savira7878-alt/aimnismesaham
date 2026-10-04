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
st.caption(f"AI Stock Screener IDX (±130 saham likuid)  •  Update: {datetime.now().strftime('%d %b %Y • %H:%M')} WIB  •  Data: Yahoo Finance  •  Bukan saran investasi")

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

st.caption("Proses scan membutuhkan 30–60 detik. Mohon tunggu sampai selesai.")

if scan:
    # Cek tren IHSG dulu
    ihsg = get_ihsg_trend()
    
    if respect_ihsg and ihsg["status"] == "Bearish":
        st.warning(f"⚠️ **IHSG sedang Bearish** (5D: {ihsg['change_5d']:+.1f}%). Rekomendasi dibatasi. Pertimbangkan tunggu market membaik.")
    
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
        smart = smart_money_score(tech)

        # Filter
        if score < min_score:
            progress.progress((i + 1) / total)
            continue
        if smart < min_smart:
            progress.progress((i + 1) / total)
            continue
        if only_uptrend and tech["price"] < tech["sma20"]:
            progress.progress((i + 1) / total)
            continue
        # Jika IHSG bearish dan user aktifkan filter, naikkan standar skor
        if respect_ihsg and ihsg["status"] == "Bearish" and score < 75:
            progress.progress((i + 1) / total)
            continue

        results.append({
            "Kode": kode,
            "Nama": fund["name"][:22],
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
        })
        progress.progress((i + 1) / total)
        time.sleep(0.10)

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
        st.caption("Urut berdasarkan Skor AI • Dilengkapi Entry Zone, SL & TP")
        
        top_cols = st.columns(min(3, len(df)))
        for idx, col in enumerate(top_cols):
            if idx < len(df):
                row = df.iloc[idx]
                with col:
                    st.metric(
                        label=f"{score_color(row['Skor AI'])} {row['Kode']}",
                        value=f"{row['Harga']:,.0f}",
                        delta=f"AI:{row['Skor AI']} | SM:{row['Smart $']}"
                    )
                    st.caption(f"Entry: {row['Entry']}")
                    st.caption(f"SL: {row['SL']:,.0f} | TP1: {row['TP1']:,.0f}")

        st.markdown("---")
        st.markdown(f"### 📋 Daftar Lengkap + Level Trading ({len(df)} saham)")

        # Format tabel
        display = df.copy()
        display["Harga"] = display["Harga"].map(lambda x: f"{x:,.0f}")
        display["1D%"] = display["1D%"].map(lambda x: f"{x:+.2f}")
        display["5D%"] = display["5D%"].map(lambda x: f"{x:+.2f}")
        display["RSI"] = display["RSI"].map(lambda x: f"{x:.1f}")
        display["Vol"] = display["Vol"].map(lambda x: f"{x:.2f}x")
        display["Skor AI"] = display["Skor AI"].map(lambda x: f"{score_color(x)} {x}")
        display["Smart $"] = display["Smart $"].map(lambda x: f"{score_color(x)} {x}")
        display["Support"] = display["Support"].map(lambda x: f"{x:,.0f}")
        display["Resistance"] = display["Resistance"].map(lambda x: f"{x:,.0f}")
        display["SL"] = display["SL"].map(lambda x: f"{x:,.0f}")
        display["TP1"] = display["TP1"].map(lambda x: f"{x:,.0f}")
        display["TP2"] = display["TP2"].map(lambda x: f"{x:,.0f}")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Kode": st.column_config.TextColumn("Kode", width="small"),
                "Nama": st.column_config.TextColumn("Nama", width="medium"),
                "Skor AI": st.column_config.TextColumn("AI", width="small"),
                "Smart $": st.column_config.TextColumn("SM", width="small"),
                "Entry": st.column_config.TextColumn("Entry Zone", width="small"),
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
    ### Fitur AimnismeSaham Sekarang
    1. **Skor AI** (Teknikal + Fundamental)
    2. **Smart Money Score** (proxy volume / bandar)
    3. **Support & Resistance** + Entry Zone
    4. **Stop Loss & Take Profit** (berdasarkan ATR)
    5. **Filter tren IHSG** (hindari open saat market bearish)

    **Cara pakai terbaik:**
    - Scan setelah jam 16:00 WIB
    - Pilih saham Skor AI ≥ 80 + Smart Money ≥ 70
    - Entry di zona yang disarankan
    - Pasang SL sesuai saran
    - Jangan force open kalau IHSG sedang Bearish
    """)

st.markdown("---")
st.caption("⚠️ Tool edukasi pribadi. Bukan rekomendasi investasi. Data bisa delay. Selalu lakukan riset mandiri.")
