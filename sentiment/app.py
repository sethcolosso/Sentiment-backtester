# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# === IMPORT YOUR BACKTESTER ===
try:
    from backtester import run_backtest
except Exception as e:
    st.error(f"Failed to import backtester.py: {e}")
    st.stop()

st.set_page_config(page_title="Backtester", layout="wide")

st.title("Market Sentiment Backtester")
st.markdown("**P/E + VIX + Yield Curve + RSI Strategy**")

# === Sidebar ===
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", datetime(2022, 5, 10))
    run = st.button("Run Backtest", type="primary")

# === Run Backtest ===
if run:
    with st.spinner("Fetching data and running backtest..."):
        try:
            df, metrics = run_backtest(ticker, start_date.strftime('%Y-%m-%d'))

            # === Results ===
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Return", f"{metrics['total_return']:+.2%}")
            col2.metric("Buy & Hold", f"{metrics['buy_hold']:+.2%}")
            col3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
            col4.metric("Max Drawdown", f"{metrics['max_dd']:.2%}")

            st.success(f"**{metrics['trades']} trades** | Signal: **{metrics['latest_signal']}**")

            # === Plot ===
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(df.index, df['Cum_Strat'], label='Strategy', color='blue')
            ax1.plot(df.index, df['Cum_Market'], label='Buy & Hold', color='gray')
            ax1.set_title(f'{ticker} Performance')
            ax1.legend()
            ax1.grid()

            ax2.plot(df.index, df['PE'], label='P/E', color='purple')
            ax2.axhline(30, color='green', linestyle='--', label='Buy < 30')
            ax2.axhline(40, color='red', linestyle='--', label='Sell > 40')
            ax2.set_title('P/E Ratio')
            ax2.legend()
            ax2.grid()

            st.pyplot(fig)

            # === Download ===
            csv = df.to_csv()
            st.download_button("Download Full Results", csv, f"{ticker}_backtest.csv", "text/csv")

        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.info("Try: AAPL, MSFT, SPY | Start date after 2015")

else:
    st.info("Enter a ticker and click **Run Backtest** to begin.")