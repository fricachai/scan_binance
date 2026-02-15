# app.py
# Binance USDT-M Perpetual「全市場掃描」Streamlit 版
# ✅ MA7 剛上穿 MA25（可限制 gap%）
# ✅ RSI > 門檻
# ✅ MACD > 0 且「最近 N 根內翻正」
# ✅ 成交量放大（當根 / 近N根均量）
# ✅ 距離 MA99 不要太遠（避免已漲一大段）
# ✅ 若雲端/地區被 Binance 擋（403/451），可改用「手動貼上幣種清單」也能跑

import os
import time
import json
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

# ========= 基本設定 =========
BASE_FAPI = os.getenv("BINANCE_FAPI_BASE", "https://fapi.binance.com")  # USDT-M Futures
DEFAULT_INTERVAL = "15m"

# 若你用中繼/代理（推薦用於雲端被擋 403/451）
# 例：export HTTPS_PROXY="http://user:pass@host:port"
HTTPS_PROXY = os.getenv("HTTPS_PROXY", "").strip()
HTTP_PROXY = os.getenv("HTTP_PROXY", "").strip()

PROXIES = None
if HTTPS_PROXY or HTTP_PROXY:
    PROXIES = {}
    if HTTP_PROXY:
        PROXIES["http"] = HTTP_PROXY
    if HTTPS_PROXY:
        PROXIES["https"] = HTTPS_PROXY

# ========= 指標 =========
def sma(x, n: int):
    return pd.Series(x).rolling(n).mean().to_numpy()

def ema(x, n: int):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()

def rsi(close, length=14):
    c = pd.Series(close)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).to_numpy()

def macd(close, fast=12, slow=26, signal=9):
    efast = ema(close, fast)
    eslow = ema(close, slow)
    macd_line = efast - eslow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().to_numpy()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def macd_flipped_recently(macd_line: np.ndarray, idx: int, lookback: int) -> bool:
    # idx 指「已收K」位置的正整數索引
    if idx <= 0 or idx >= len(macd_line):
        return False
    if np.isnan(macd_line[idx]) or macd_line[idx] <= 0:
        return False
    start = idx - lookback
    if start < 0:
        return False
    window = macd_line[start:idx + 1]
    if np.isnan(window).any():
        return False
    return np.min(window) <= 0

# ========= HTTP / Binance =========
def http_get(url, params=None, timeout=20, retries=2, sleep_sec=0.6):
    """
    Streamlit Cloud 會 redacted 例外細節，所以這裡回傳 (ok, status_code, text, final_url)
    你可以在畫面直接看到 403 / 451 等狀態碼
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; scan_binance/1.0)",
        "Accept": "application/json",
    }

    last_status = None
    last_text = ""
    last_url = url

    for _ in range(retries + 1):
        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                proxies=PROXIES,
            )
            last_status = r.status_code
            last_url = r.url
            if r.status_code == 200:
                return True, r.status_code, r.text, r.url

            # 非 200：記錄一小段，避免太長
            last_text = (r.text or "")[:800]
            time.sleep(sleep_sec)

        except Exception as e:
            last_status = "EXC"
            last_text = repr(e)
            time.sleep(sleep_sec)

    return False, last_status, last_text, last_url

def get_json(url, params=None, timeout=20, retries=2):
    ok, code, text, final_url = http_get(url, params=params, timeout=timeout, retries=retries)
    if not ok:
        return None, code, text, final_url
    try:
        return json.loads(text), code, "", final_url
    except Exception as e:
        return None, "JSON_ERR", f"{repr(e)}\n{text[:400]}", final_url

@st.cache_data(ttl=60 * 30, show_spinner=False)
def list_usdt_perp_symbols_cached(base_fapi: str):
    """
    先試 exchangeInfo（最完整），失敗再試 premiumIndex（通常更容易）
    若兩者都被擋，回傳空陣列，讓使用者用手動貼上清單
    """
    # 1) exchangeInfo
    j, code, err, u = get_json(f"{base_fapi}/fapi/v1/exchangeInfo", timeout=20, retries=2)
    if j and isinstance(j, dict) and "symbols" in j:
        syms = []
        for s in j["symbols"]:
            if s.get("contractType") != "PERPETUAL":
                continue
            if s.get("quoteAsset") != "USDT":
                continue
            if s.get("status") != "TRADING":
                continue
            syms.append(s["symbol"])
        syms = sorted(set(syms))
        return syms, ("exchangeInfo", 200, "")

    # 2) premiumIndex（會回所有目前有標記的合約）
    j2, code2, err2, u2 = get_json(f"{base_fapi}/fapi/v1/premiumIndex", timeout=20, retries=2)
    if j2 and isinstance(j2, list):
        syms = []
        for it in j2:
            s = it.get("symbol", "")
            # 排除指數/組合等（通常含 "_"）
            if s.endswith("USDT") and "_" not in s:
                syms.append(s)
        syms = sorted(set(syms))
        return syms, ("premiumIndex", 200, "")

    # 都失敗
    detail = f"exchangeInfo: code={code}, err={err}\npremiumIndex: code={code2}, err={err2}"
    return [], ("FAILED", code2, detail)

@st.cache_data(ttl=60 * 10, show_spinner=False)
def get_klines_cached(base_fapi: str, symbol: str, interval: str, limit: int):
    j, code, err, u = get_json(
        f"{base_fapi}/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": int(limit)},
        timeout=20,
        retries=2
    )
    return j, code, err, u

# ========= 核心檢查 =========
def check_symbol(
    base_fapi: str,
    symbol: str,
    interval: str,
    limit: int,
    ma_fast: int,
    ma_slow: int,
    ma_long: int,
    rsi_len: int,
    rsi_min: float,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    vol_ma_len: int,
    vol_mult: float,
    gap_max: float,
    dist99_min: float,
    dist99_max: float,
    macd_flip_lookback: int,
    require_hist_rising: bool,
):
    ks, code, err, _ = get_klines_cached(base_fapi, symbol, interval, limit)
    if ks is None or not isinstance(ks, list) or len(ks) < max(ma_long, macd_slow + macd_signal) + 10:
        return None, f"{symbol}: klines failed code={code} {err}"

    close = np.array([float(k[4]) for k in ks], dtype=float)
    volume = np.array([float(k[5]) for k in ks], dtype=float)
    close_time = np.array([int(k[6]) for k in ks], dtype=np.int64)

    ma7 = sma(close, ma_fast)
    ma25 = sma(close, ma_slow)
    ma99 = sma(close, ma_long)

    r = rsi(close, rsi_len)
    macd_line, sig, hist = macd(close, macd_fast, macd_slow, macd_signal)
    vol_ma = sma(volume, vol_ma_len)

    # 取上一根已收K（避免用未收K）
    i = len(close) - 2
    need = [ma7, ma25, ma99, r, macd_line, hist, vol_ma]
    for arr in need:
        if i < 1 or i >= len(arr) or np.isnan(arr[i]) or np.isnan(arr[i - 1]):
            return None, None

    cross_up = (ma7[i - 1] <= ma25[i - 1]) and (ma7[i] > ma25[i])
    ma7_rising = ma7[i] > ma7[i - 1]

    gap = (ma7[i] - ma25[i]) / ma25[i]
    dist99 = (close[i] - ma99[i]) / ma99[i]

    rsi_ok = r[i] > rsi_min
    macd_ok = macd_line[i] > 0

    hist_rising = (hist[i] > hist[i - 1]) if require_hist_rising else True

    vol_ratio = (volume[i] / vol_ma[i]) if vol_ma[i] and not np.isnan(vol_ma[i]) else np.nan
    vol_ok = (not np.isnan(vol_ratio)) and (vol_ratio >= vol_mult)

    macd_flip_ok = macd_flipped_recently(macd_line, i, macd_flip_lookback)

    ok = (
        cross_up
        and ma7_rising
        and (gap <= gap_max)
        and (dist99_min <= dist99 <= dist99_max)
        and rsi_ok
        and macd_ok
        and hist_rising
        and macd_flip_ok
        and vol_ok
    )

    if not ok:
        return None, None

    ts = datetime.fromtimestamp(close_time[i] / 1000, tz=timezone.utc).astimezone()
    return {
        "symbol": symbol,
        "time_local": ts.strftime("%Y-%m-%d %H:%M:%S %z"),
        "close": close[i],
        "ma_fast": ma7[i],
        "ma_slow": ma25[i],
        "ma_long": ma99[i],
        "gap_%": round(gap * 100, 3),
        "dist99_%": round(dist99 * 100, 3),
        "rsi": round(float(r[i]), 2),
        "macd": float(macd_line[i]),
        "hist": float(hist[i]),
        "vol_ratio": round(float(vol_ratio), 2),
    }, None

# ========= Streamlit UI =========
def main():
    st.set_page_config(page_title="Binance USDT-M Perp Scanner", layout="wide")
    st.title("Binance USDT-M 永續合約掃描（MA7 上穿 MA25 + RSI/MACD/量能/MA99）")

    with st.sidebar:
        st.header("資料來源")
        base_fapi = st.text_input("Binance Futures API Base", value=BASE_FAPI)
        interval = st.selectbox("K線週期", ["1m", "5m", "15m", "30m", "1h", "4h"], index=2)
        limit = st.slider("抓取K線根數（越多越慢）", 200, 1500, 500, 50)

        st.divider()
        st.header("均線條件")
        ma_fast = st.number_input("MA 快線", 2, 50, 7, 1)
        ma_slow = st.number_input("MA 慢線", 5, 200, 25, 1)
        ma_long = st.number_input("MA 長線（避免已漲太多）", 20, 400, 99, 1)

        gap_max_pct = st.slider("剛突破上方距離上限（gap%）", 0.1, 5.0, 2.0, 0.1)
        dist99_min_pct = st.slider("相對 MA99 下限（%）", -30.0, 0.0, -5.0, 0.5)
        dist99_max_pct = st.slider("相對 MA99 上限（%）", 0.0, 30.0, 8.0, 0.5)

        st.divider()
        st.header("RSI / MACD")
        rsi_len = st.number_input("RSI 長度", 5, 50, 14, 1)
        rsi_min = st.slider("RSI 下限", 0.0, 80.0, 50.0, 1.0)

        macd_fast = st.number_input("MACD fast", 2, 50, 12, 1)
        macd_slow = st.number_input("MACD slow", 5, 100, 26, 1)
        macd_signal = st.number_input("MACD signal", 2, 50, 9, 1)

        macd_flip_lookback = st.slider("MACD 最近 N 根內翻正", 1, 80, 16, 1)
        require_hist_rising = st.checkbox("要求 hist 上升（更偏確認）", value=True)

        st.divider()
        st.header("量能")
        vol_ma_len = st.number_input("均量視窗（根）", 5, 200, 20, 1)
        vol_mult = st.slider("當根成交量 / 均量 ≥", 1.0, 5.0, 1.3, 0.1)

        st.divider()
        st.header("掃描範圍")
        max_symbols = st.slider("最多掃描幣數（避免太慢/被限速）", 50, 800, 450, 50)

        st.caption("若雲端被擋（403/451），請改用下方「手動貼幣種清單」或使用 HTTPS_PROXY。")

    # 取得幣清單（快取）
    syms, meta = list_usdt_perp_symbols_cached(base_fapi)
    src, code, detail = meta

    colA, colB = st.columns([1.2, 1.0], gap="large")

    with colA:
        st.subheader("1) 幣種清單")
        if syms:
            st.success(f"取得幣種成功：{len(syms)}（來源：{src}）")
        else:
            st.error("無法從 Binance 取得幣種清單（常見原因：403/451 被擋）")
            st.code(detail)

        manual = st.text_area(
            "手動貼上幣種清單（每行一個，例如 BTCUSDT）— 當自動取得失敗時用這個",
            value="",
            height=160
        )

        if manual.strip():
            manual_syms = []
            for line in manual.splitlines():
                s = line.strip().upper()
                if not s:
                    continue
                if s.endswith("USDT") and "_" not in s:
                    manual_syms.append(s)
            manual_syms = sorted(set(manual_syms))
            st.info(f"手動清單：{len(manual_syms)}")
            symbols = manual_syms
        else:
            symbols = syms

        if symbols:
            st.caption("（預覽前 50 支）")
            st.code("\n".join(symbols[:50]))
        else:
            st.stop()

    with colB:
        st.subheader("2) 立即掃描")
        start = st.button("開始掃描", type="primary", use_container_width=True)

        st.markdown("**目前條件摘要**")
        st.write(
            {
                "interval": interval,
                "MA": f"{ma_fast}/{ma_slow}/{ma_long}",
                "gap_max_%": gap_max_pct,
                "dist99_%": f"{dist99_min_pct} ~ {dist99_max_pct}",
                "RSI>": rsi_min,
                "MACD>0 且最近翻正N": macd_flip_lookback,
                "hist上升": require_hist_rising,
                "vol_ratio>=": vol_mult,
            }
        )

    if not start:
        st.stop()

    # 限制掃描數
    scan_syms = symbols[: int(max_symbols)]

    st.subheader("3) 掃描結果")
    prog = st.progress(0)
    status = st.empty()

    hits = []
    errs = 0

    # 轉成比例
    gap_max = gap_max_pct / 100.0
    dist99_min = dist99_min_pct / 100.0
    dist99_max = dist99_max_pct / 100.0

    # 掃描
    for idx, sym in enumerate(scan_syms, 1):
        status.write(f"掃描中：{idx}/{len(scan_syms)}  {sym}")
        row, err = check_symbol(
            base_fapi=base_fapi,
            symbol=sym,
            interval=interval,
            limit=int(limit),
            ma_fast=int(ma_fast),
            ma_slow=int(ma_slow),
            ma_long=int(ma_long),
            rsi_len=int(rsi_len),
            rsi_min=float(rsi_min),
            macd_fast=int(macd_fast),
            macd_slow=int(macd_slow),
            macd_signal=int(macd_signal),
            vol_ma_len=int(vol_ma_len),
            vol_mult=float(vol_mult),
            gap_max=float(gap_max),
            dist99_min=float(dist99_min),
            dist99_max=float(dist99_max),
            macd_flip_lookback=int(macd_flip_lookback),
            require_hist_rising=bool(require_hist_rising),
        )

        if row:
            hits.append(row)

        if err:
            errs += 1

        prog.progress(idx / len(scan_syms))
        # 小延遲避免 API 被限速（可自行調小/調大）
        time.sleep(0.06)

    status.empty()

    if not hits:
        st.warning("沒有找到符合條件的幣種。你可以：放寬 gap%、降低 vol_ratio、放寬 dist99 上限、或把 hist 上升取消。")
        st.info(f"掃描 {len(scan_syms)} 支；klines 失敗/被擋筆數（估計）：{errs}")
        st.stop()

    df = pd.DataFrame(hits).sort_values(["time_local", "vol_ratio"], ascending=[False, False]).reset_index(drop=True)
    st.success(f"命中：{len(df)} 支（掃描 {len(scan_syms)} 支；klines 失敗/被擋估計：{errs}）")

    st.dataframe(df, use_container_width=True, height=520)

    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下載結果 CSV", data=csv, file_name="signals.csv", mime="text/csv")

    st.caption("提示：若你在 Streamlit Cloud 仍被 403/451，請改：本機跑 / VPS（台灣）跑 / 或設定 HTTPS_PROXY（或做中繼 API）。")

if __name__ == "__main__":
    main()
