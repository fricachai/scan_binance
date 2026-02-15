import time
import requests
import pandas as pd
import numpy as np

BASE_FAPI = "https://fapi.binance.com"  # USDT-M Futures

# ===== 參數（你可以直接調）=====
INTERVAL = "15m"

MA_FAST = 7
MA_SLOW = 25
MA_LONG = 99

RSI_LEN = 14
RSI_MIN = 50.0

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

VOL_MA_LEN = 20
VOL_MULT_A = 1.5     # A級更嚴格
VOL_MULT_B = 1.3     # B級稍放寬

# 「剛突破」幅度（gap = (ma7-ma25)/ma25）
GAP_MAX_A = 0.010    # 1.0%
GAP_MAX_B = 0.020    # 2.0%

# 距離MA99
DIST99_MIN = -0.05   # -5%
DIST99_MAX_A = 0.05  # +5%  (A級)
DIST99_MAX_B = 0.08  # +8%  (B級)

# MACD 「最近N根內翻正」
MACD_FLIP_LOOKBACK_A = 8    # 2小時內翻正（15m*8=120m）
MACD_FLIP_LOOKBACK_B = 16   # 4小時內翻正（15m*16=240m）

OUT_A = "signals_A_super_fresh.csv"
OUT_B = "signals_B_confirmed.csv"

# K 線根數（要能算到 MA99 + MACD 26）
KLINES_LIMIT = 260  # >= 99 + 26 + buffer


# ===== HTTP 共用：加 header + retry =====
def http_get(url, params=None, timeout=20, retries=2, sleep_sec=0.6):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; scan_binance/1.0)",
        "Accept": "application/json",
    }
    last_exc = None
    for _ in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # 直接在這裡檢查狀態碼，讓錯誤訊息更清楚
            if r.status_code != 200:
                raise requests.HTTPError(
                    f"HTTP {r.status_code} for {r.url}\nResponse: {r.text[:300]}"
                )
            return r
        except Exception as e:
            last_exc = e
            time.sleep(sleep_sec)
    raise last_exc


# ===== 指標 =====
def sma(x, n):
    return pd.Series(x).rolling(n).mean().to_numpy()

def ema(x, n):
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


# ===== Futures API（改：不再用 exchangeInfo）=====
def list_usdt_perp_symbols():
    """
    用 premiumIndex 取所有永續合約（此端點回傳的本來就是 PERPETUAL）
    再篩出 USDT 計價，且排除交割/特殊格式（通常含 '_'）
    """
    url = f"{BASE_FAPI}/fapi/v1/premiumIndex"
    r = http_get(url, timeout=20, retries=2)
    data = r.json()

    syms = []
    for it in data:
        s = it.get("symbol", "")
        if not s.endswith("USDT"):
            continue
        if "_" in s:  # 排除交割合約（例如 BTCUSDT_240329）
            continue
        syms.append(s)

    return sorted(set(syms))

def get_klines(symbol, interval="15m", limit=260):
    url = f"{BASE_FAPI}/fapi/v1/klines"
    r = http_get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": int(limit)},
        timeout=20,
        retries=2
    )
    return r.json()


# ===== MACD 最近N根內翻正 =====
def macd_flipped_recently(macd_line, i, lookback):
    # macd_line[i] > 0 且 lookback 範圍內存在 <=0
    if macd_line[i] <= 0:
        return False
    start = i - lookback
    if start < 0:
        return False
    window = macd_line[start:i+1]
    return np.nanmin(window) <= 0


# ===== 檢查單一幣（回傳 A/B 訊號）=====
def check_symbol(symbol):
    ks = get_klines(symbol, INTERVAL, KLINES_LIMIT)
    if len(ks) < 120:
        return (None, None)

    close = np.array([float(k[4]) for k in ks], dtype=float)
    volume = np.array([float(k[5]) for k in ks], dtype=float)
    close_time = np.array([int(k[6]) for k in ks], dtype=np.int64)

    ma7 = sma(close, MA_FAST)
    ma25 = sma(close, MA_SLOW)
    ma99 = sma(close, MA_LONG)

    r = rsi(close, RSI_LEN)
    macd_line, sig, hist = macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    vol_ma = sma(volume, VOL_MA_LEN)

    i = -2  # 上一根已收K

    # 檢查必要指標是否可用
    if any(np.isnan(arr[i]) for arr in [ma7, ma25, ma99, r, macd_line, hist, vol_ma]):
        return (None, None)

    # 核心交叉（剛突破）
    cross_up = (ma7[i-1] <= ma25[i-1]) and (ma7[i] > ma25[i])
    ma7_rising = ma7[i] > ma7[i-1]

    gap = (ma7[i] - ma25[i]) / ma25[i]
    dist99 = (close[i] - ma99[i]) / ma99[i]

    rsi_ok = r[i] > RSI_MIN
    macd_ok = macd_line[i] > 0

    # 動能仍在增強（B級用）
    hist_rising = hist[i] > hist[i-1]

    # 量能
    vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] != 0 else np.nan

    # 目前這根（i=-2）換成實際 index（正索引）
    i_abs = len(macd_line) + i  # 等於 len-2

    # A級：超剛、訊號少、最早
    A = (
        cross_up and ma7_rising and
        (gap <= GAP_MAX_A) and
        (DIST99_MIN <= dist99 <= DIST99_MAX_A) and
        rsi_ok and macd_ok and
        macd_flipped_recently(macd_line, i_abs, MACD_FLIP_LOOKBACK_A) and
        (vol_ratio >= VOL_MULT_A)
    )

    # B級：剛啟動但更實戰（較多訊號、較穩）
    B = (
        cross_up and ma7_rising and
        (gap <= GAP_MAX_B) and
        (DIST99_MIN <= dist99 <= DIST99_MAX_B) and
        rsi_ok and macd_ok and hist_rising and
        macd_flipped_recently(macd_line, i_abs, MACD_FLIP_LOOKBACK_B) and
        (vol_ratio >= VOL_MULT_B)
    )

    base = {
        "symbol": symbol,
        "time": pd.to_datetime(close_time[i], unit="ms"),
        "close": float(close[i]),
        "gap_%": round(float(gap) * 100, 3),
        "dist99_%": round(float(dist99) * 100, 3),
        "rsi": round(float(r[i]), 2),
        "macd": float(macd_line[i]),
        "hist": float(hist[i]),
        "vol_ratio": round(float(vol_ratio), 2)
    }

    return (base if A else None, base if B else None)


def main():
    # 取 USDT-M 永續清單（不再用 exchangeInfo）
    symbols = list_usdt_perp_symbols()
    print(f"Scanning Futures PERP (via premiumIndex) symbols: {len(symbols)}")

    hits_A, hits_B = [], []

    for idx, sym in enumerate(symbols, 1):
        try:
            a, b = check_symbol(sym)
            if a:
                hits_A.append(a)
                print(f"[A HIT] {sym} gap={a['gap_%']}% dist99={a['dist99_%']}% volx={a['vol_ratio']}")
            if b:
                hits_B.append(b)
                print(f"[B HIT] {sym} gap={b['gap_%']}% dist99={b['dist99_%']}% volx={b['vol_ratio']}")
        except Exception as e:
            # 建議你先別完全吞錯，至少每 50 支印一次，方便抓「哪個 endpoint 被擋」
            if idx % 50 == 0:
                print(f"[WARN] example error @ {sym}: {repr(e)}")
            pass

        time.sleep(0.08)
        if idx % 150 == 0:
            print(f"Progress {idx}/{len(symbols)}")

    if hits_A:
        dfA = pd.DataFrame(hits_A).sort_values(["time", "vol_ratio"], ascending=[False, False])
        dfA.to_csv(OUT_A, index=False, encoding="utf-8-sig")
        print(f"\nSaved A signals: {OUT_A}  (count={len(dfA)})")
        print(dfA.head(30).to_string(index=False))
    else:
        print("\nNo A signals found.")

    if hits_B:
        dfB = pd.DataFrame(hits_B).sort_values(["time", "vol_ratio"], ascending=[False, False])
        dfB.to_csv(OUT_B, index=False, encoding="utf-8-sig")
        print(f"\nSaved B signals: {OUT_B}  (count={len(dfB)})")
        print(dfB.head(30).to_string(index=False))
    else:
        print("\nNo B signals found.")


if __name__ == "__main__":
    main()
