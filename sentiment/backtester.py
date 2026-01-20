# backtester.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import warnings
import time
warnings.filterwarnings("ignore")

SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

# === FETCHS FUNCTIONS ===
def fetch_price(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    for _ in range(3):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty: raise ValueError("Empty")
            data = data[['Close']].copy()
            data.rename(columns={'Close': 'Price'}, inplace=True)
            return data
        except Exception as e:
            time.sleep(3)
    raise ValueError("Price fetch failed")

def fetch_earnings(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.quarterly_earnings
        if earnings is None or earnings.empty: raise ValueError("No data")
        earnings = earnings.copy()
        earnings.index = pd.to_datetime(earnings.index)
        earnings = earnings[['Earnings']]
        full_range = pd.date_range(start=start, end=end, freq='D')
        daily = earnings.reindex(full_range).ffill()
        return daily
    except:
        try:
            eps = stock.info.get('trailingEps', 6.0)
            series = pd.Series(eps, index=pd.date_range(start=start, end=end, freq='D'))
            return series
        except:
            return None

def fetch_shiller_cape(start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        response = requests.get(SHILLER_URL, timeout=10)
        df = pd.read_excel(BytesIO(response.content), sheet_name="Data", skiprows=7)
        df = df.iloc[:, [0, 7]]
        df.columns = ['Date', 'CAPE']
        df['Date'] = pd.to_datetime(df['Date'].astype(str).str.split('.').str[0] + '-01')
        df = df.set_index('Date').dropna()
        full = pd.date_range(start=start, end=end, freq='D')
        cape = df['CAPE'].reindex(full).ffill()
        return cape
    except:
        return None

def fetch_vix(start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        vix = yf.download("^VIX", start=start, end=end, progress=False)['Close']
        vix.name = 'VIX'
        return vix
    except:
        return pd.Series(20.0, index=pd.date_range(start=start, end=end, freq='D'))

def fetch_yield_curve(start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    try:
        t10 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10", parse_dates=['DATE'])
        t2 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2", parse_dates=['DATE'])
        yc = t10.set_index('DATE')['DGS10'] - t2.set_index('DATE')['DGS2']
        yc = yc.reindex(pd.date_range(start=start, end=end, freq='D')).ffill()
        yc.name = 'Yield_Curve'
        return yc
    except:
        return pd.Series(0.5, index=pd.date_range(start=start, end=end, freq='D'))

# === SIGNALS & BACKTESTING ===

def generate_signal(pe, vix, yc, rsi):
    score = 0
    if pe < 30: score += 1
    elif pe > 40: score -= 1
    if vix < 18: score += 1
    elif vix > 30: score -= 1
    if yc > 0.3: score += 1
    if rsi < 30: score += 1
    if rsi > 70: score -= 1
    if score >= 2: return "BUY"
    elif score <= -2: return "SELL"
    else: return "HOLD"

def run_backtest(ticker, start, end=None):
    end = end or datetime.today().strftime('%Y-%m-%d')
    price = fetch_price(ticker, start, end)
    vix = fetch_vix(start, end)
    yc = fetch_yield_curve(start, end)

    try:
        eps = yf.Ticker(ticker).info.get('trailingEps', 6.0)
    except:
        eps = 6.0

    df = price.copy()
    df['EPS'] = eps
    df['PE'] = df['Price'] / eps
    df['VIX'] = vix.reindex(df.index).ffill().bfill()
    df['Yield_Curve'] = yc.reindex(df.index).ffill().bfill()

    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)

    df['Signal'] = 'HOLD'
    for i in range(len(df)):
        df.iloc[i, df.columns.get_loc('Signal')] = generate_signal(
            df['PE'].iloc[i], df['VIX'].iloc[i], df['Yield_Curve'].iloc[i], df['RSI'].iloc[i]
        )

    df['Return'] = df['Price'].pct_change().fillna(0)
    df['Strategy'] = 0.0
    in_pos = False
    cost = 0.001
    trades = 0

    for i in range(1, len(df)):
        sig = df['Signal'].iloc[i-1]
        ret = df['Return'].iloc[i]
        strat = 0.0
        if sig == "BUY" and not in_pos:
            in_pos = True
            trades += 1
            strat = ret - cost
        elif sig == "SELL" and in_pos:
            in_pos = False
            trades += 1
            strat = ret - cost
        elif in_pos:
            strat = ret
            if ret < -0.05:
                strat -= cost
                in_pos = False
                trades += 1
        df.iloc[i, df.columns.get_loc('Strategy')] = strat

    df['Cum_Strat'] = (1 + df['Strategy']).cumprod()
    df['Cum_Market'] = (1 + df['Return']).cumprod()

    total = df['Cum_Strat'].iloc[-1] - 1
    market = df['Cum_Market'].iloc[-1] - 1
    days = len(df)
    ann = (1 + total) ** (252/days) - 1 if days else 0
    vol = df['Strategy'].std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0
    dd = (df['Cum_Strat'] / df['Cum_Strat'].cummax() - 1).min()

    metrics = {
        'total_return': total,
        'buy_hold': market,
        'annualized': ann,
        'sharpe': sharpe,
        'max_dd': dd,
        'trades': trades,
        'latest_signal': df['Signal'].iloc[-1],
        'latest_pe': df['PE'].iloc[-1],
        'latest_rsi': df['RSI'].iloc[-1]
    }
    return df, metrics
