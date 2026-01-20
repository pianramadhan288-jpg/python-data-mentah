import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

def analyze_ticker(ticker, benchmark_ticker='^JKSE'):
    try:
        yf_data = yf.Ticker(ticker)
        data = yf_data.history(period='max')
        info = yf_data.info

        yf_bench = yf.Ticker(benchmark_ticker)
        bench_data = yf_bench.history(period='max')

        common_dates = data.index.intersection(bench_data.index)
        data = data.loc[common_dates]
        bench_data = bench_data.loc[common_dates]

        if data.empty:
            st.error("Data kosong. Cek ticker benar?")
            return

        # 1. Fundamental Komplit
        st.subheader("=== Data Fundamental Komplit ===")
        st.write(f"Ticker: {ticker}")
        st.write(f"Currency: {info.get('currency', 'N/A')}")
        st.write(f"Harga Close Terbaru: {data['Close'].iloc[-1]:.2f}")
        st.write(f"Market Cap: {info.get('marketCap', 'N/A')}")
        st.write(f"Trailing PE Ratio: {info.get('trailingPE', 'N/A')}")
        st.write(f"Forward PE Ratio: {info.get('forwardPE', 'N/A')}")
        st.write(f"Trailing EPS: {info.get('trailingEps', 'N/A')}")
        st.write(f"Forward EPS: {info.get('forwardEps', 'N/A')}")
        st.write(f"PEG Ratio: {info.get('pegRatio', 'N/A')}")
        st.write(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
        st.write(f"Beta: {info.get('beta', 'N/A')}")
        st.write(f"Target Mean Price: {info.get('targetMeanPrice', 'N/A')}")
        st.write(f"Target High/Low: High {info.get('targetHighPrice', 'N/A')} Low {info.get('targetLowPrice', 'N/A')}")
        st.write(f"Number of Analysts: {info.get('numberOfAnalystOpinions', 'N/A')}")

        # 2. Data Historis 5 Hari (teks polos, mudah copy)
        st.subheader("=== Data Historis 5 Hari Terakhir ===")
        recent_5 = data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].round(2)

        # Tampil sebagai teks polos mirip print di terminal
        st.markdown("**Output teks polos (mudah disalin ke Excel/Notes):**")
        text_output = recent_5.to_string()
        st.code(text_output, language="text")

        # Opsional: tombol download CSV kalau lo mau file
        csv = recent_5.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download 5 Hari sebagai CSV (opsional)",
            data=csv,
            file_name=f"{ticker}_5hari_terakhir.csv",
            mime="text/csv",
        )

        # Ringkasan 1 Bulan
        st.subheader("=== Ringkasan Historis 1 Bulan Terakhir ===")
        one_month_ago = data.index[-1] - pd.Timedelta(days=30)
        month_data = data.loc[one_month_ago:]
        st.write(f"Mean Close 1 Bulan: {month_data['Close'].mean():.2f}")
        st.write(f"Max Close 1 Bulan: {month_data['Close'].max():.2f}")
        st.write(f"Min Close 1 Bulan: {month_data['Close'].min():.2f}")
        st.write(f"Total Volume 1 Bulan: {month_data['Volume'].sum()}")
        st.write(f"Rata-rata Volume Harian 1 Bulan: {month_data['Volume'].mean():.0f}")

        # 3. Stats Matematis Komplit
        returns = data['Close'].pct_change().dropna()
        bench_returns = bench_data['Close'].pct_change().dropna()
        st.subheader("=== Stats Matematis Komplit ===")
        st.write(f"Mean Return Harian: {returns.mean():.6f}")
        st.write(f"Std Dev Return (Volatilitas Harian): {returns.std():.6f}")
        st.write(f"Sharpe Ratio Sederhana (Risk-Free 0): {(returns.mean() / returns.std()) if returns.std() != 0 else 'N/A'}")

        annual_return = (1 + returns.mean()) ** 252 - 1
        st.write(f"Annual Return Estimasi: {annual_return:.4f}")

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        st.write(f"Max Drawdown Historis: {max_drawdown:.4f}")

        acf_vals = acf(returns.dropna(), nlags=5, fft=False)
        st.write(f"Autocorrelation Return Lag 1-5: {acf_vals[1:6]}")

        vol_clust = returns.rolling(20).std()
        st.write(f"Rata-rata Volatility Clustering (Rolling Std 20 Hari): {vol_clust.mean():.6f}")
        st.write(f"Max Volatility Clustering: {vol_clust.max():.6f}")
        st.write(f"Min Volatility Clustering: {vol_clust.min():.6f}")

        st.write(f"Return Skewness: {returns.skew():.4f}")
        st.write(f"Return Kurtosis: {returns.kurtosis():.4f}")

        correlation = returns.corr(bench_returns)
        st.write(f"Correlation dengan Benchmark IHSG (^JKSE): {correlation:.4f}")

        momentum_score = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].rolling(10).std().iloc[-1] if len(data) > 10 else 'N/A'
        st.write(f"Momentum Score (Inersia Fisika 10 Hari): {momentum_score:.4f} (>2 kuat buy, < -2 kuat sell)")

        lambda_ewma = 0.94
        vol_forecast = np.sqrt(lambda_ewma * returns.iloc[-1]**2 + (1 - lambda_ewma) * returns.rolling(20).var().iloc[-1]) if len(returns) > 20 else 'N/A'
        st.write(f"Vol Forecast Besok (EWMA Fisika): {vol_forecast:.4f}")

        # 4. Technical Indicators
        data['MA10'] = data['Close'].rolling(10).mean()
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        data['RSI14'] = 100 - (100 / (1 + rs))
        data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
        data['BB_Mid'] = data['Close'].rolling(20).mean()
        data['BB_Std'] = data['Close'].rolling(20).std()
        data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']

        st.subheader("=== Technical Indicators Terbaru ===")
        st.write(f"MA10: {data['MA10'].iloc[-1]:.2f}")
        st.write(f"MA20: {data['MA20'].iloc[-1]:.2f}")
        st.write(f"MA50: {data['MA50'].iloc[-1]:.2f}")
        st.write(f"RSI14: {data['RSI14'].iloc[-1]:.2f}")
        st.write(f"MACD: {data['MACD'].iloc[-1]:.2f}")
        st.write(f"Bollinger Upper: {data['BB_Upper'].iloc[-1]:.2f}")
        st.write(f"Bollinger Mid: {data['BB_Mid'].iloc[-1]:.2f}")
        st.write(f"Bollinger Lower: {data['BB_Lower'].iloc[-1]:.2f}")

        data['MA_Cross'] = np.where(data['MA10'] > data['MA20'], 1, 0)
        data['MA_Cross_Shift'] = data['MA_Cross'].shift(1)
        crossovers = data[data['MA_Cross'] != data['MA_Cross_Shift']]
        st.write(f"Jumlah MA10/MA20 Crossover Historis: {len(crossovers)}")
        st.write(f"Buy Signals (MA10 > MA20): {data['MA_Cross'].sum()}")
        st.write(f"Buy Signal Terbaru: {'Yes' if data['MA_Cross'].iloc[-1] == 1 else 'No'}")

        # 5. Monte Carlo Simulasi (dengan fix empty slice)
        st.subheader("=== Monte Carlo Simulasi Komplit ===")
        sim_returns = np.random.normal(returns.mean(), returns.std(), (365, 200))
        sim_paths = (1 + sim_returns).cumprod(axis=0) * data['Close'].iloc[-1]

        sim_paths = np.nan_to_num(sim_paths, nan=0, posinf=sim_paths.max(), neginf=sim_paths.min())

        mean_path = sim_paths.mean(axis=1)[-1] if sim_paths.size > 0 else data['Close'].iloc[-1]

        last_day = sim_paths[-1, :] if sim_paths.shape[0] > 0 else np.array([data['Close'].iloc[-1]])

        var_95 = np.percentile(last_day, 5) if last_day.size > 0 else data['Close'].iloc[-1]

        worst_cases = last_day[last_day <= var_95]
        cvar_95 = worst_cases.mean() if len(worst_cases) > 0 else var_95

        st.write(f"Mean Harga Simulasi 1 Tahun: {mean_path:.2f}")
        st.write(f"VaR 95% (Worst-case 5%): {var_95:.2f}")
        st.write(f"CVaR 95% (Rata-Rata Worst-case): {cvar_95:.2f}")
        st.write(f"Simulasi Return Skewness: {np.skew(last_day):.4f}")
        st.write(f"Simulasi Return Kurtosis: {np.kurtosis(last_day):.4f}")

        autocorr_lag1 = acf_vals[1] if len(acf_vals) > 1 else 0
        theta = -np.log(autocorr_lag1) if autocorr_lag1 > 0 else 0.5
        long_term_mean = data['Close'].rolling(200).mean().iloc[-1] if len(data) > 200 else data['Close'].mean()
        ou_drift = theta * (long_term_mean - data['Close'].iloc[-1])
        ou_multiplier = np.exp(ou_drift * 1)
        ou_price_1y = data['Close'].iloc[-1] * ou_multiplier
        st.write(f"OU Mean Price 1 Tahun (Mean Reversion Fisika): {ou_price_1y:.2f}")

        fig, ax = plt.subplots(figsize=(14,7))
        ax.plot(sim_paths, alpha=0.1)
        ax.set_title(f'Monte Carlo Simulasi Harga 1 Tahun {ticker}')
        ax.set_xlabel('Hari')
        ax.set_ylabel('Harga Simulasi')
        ax.grid(True)
        st.pyplot(fig)

        st.success("=== Analisa Selesai 100% Komplit! ===")
        st.info("Semua hitungan sudah dilakukan. Copy output ini untuk olah di AI universal.")

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Data sudah cukup untuk analisa dasar. Copy yang keluar saja.")

# UI Streamlit
st.title("Quant Analisa Saham Komplit Kompleks V2 Fisika - Pian's Tool 📈")
st.markdown("Masukkan ticker (contoh: EMDE.JK) untuk analisa lengkap. Data 5 hari ditampilkan sebagai teks polos agar mudah disalin.")

ticker = st.text_input("Ticker Saham:", "EMDE.JK").strip().upper()

if st.button("Jalankan Analisa"):
    with st.spinner("Mengambil data & menghitung... (bisa 10-30 detik)"):
        analyze_ticker(ticker)

st.markdown("---")
st.info("App ini ringan dan fokus pada copy-paste mudah. Jalankan ulang dengan ticker baru.")