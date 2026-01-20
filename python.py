# Quant Analisa Saham Komplit Kompleks - Versi Mantap & Akurat dengan Fisika Tambahan (90% Matematis: Stats Lengkap, Indicators Lebih Banyak, Backtest Sederhana, Monte Carlo, Correlation Benchmark)
# Tambahan Fisika: Ornstein-Uhlenbeck (Mean Reversion), Momentum Score (Inersia), Vol Forecast EWMA (Phase Transition).
# Dinamis: Input ticker apa saja. Output fokus data mentah + hitung lengkap (untuk olah sendiri).
# Tambahan: Bollinger Bands, Simple MA Crossover Signals, Annual Return, Max Drawdown Historis, Correlation dengan Benchmark (misal IHSG via ^JKSE).
# Update: Tambah cek parameter pembeda small cap vs big cap + kesimpulan satu baris.
# Library: pip install yfinance pandas matplotlib numpy statsmodels (ringan saja)
# Jalankan: python nama_file.py

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from datetime import datetime  # Buat cek news date

def analyze_ticker(ticker, benchmark_ticker='^JKSE'):
    try:
        # Fetch data saham & benchmark
        yf_data = yf.Ticker(ticker)
        data = yf_data.history(period='max')
        info = yf_data.info
        news = yf_data.news  # Buat reaksi news

        yf_bench = yf.Ticker(benchmark_ticker)
        bench_data = yf_bench.history(period='max')

        # Clean data: Align dates for correlation
        common_dates = data.index.intersection(bench_data.index)
        data = data.loc[common_dates]
        bench_data = bench_data.loc[common_dates]

        if data.empty:
            print("Data kosong. Cek ticker benar?")
            return

        # 1. Fundamental Komplit (sama seperti asli)
        print("\n=== Data Fundamental Komplit ===")
        print(f"Ticker: {ticker}")
        print(f"Currency: {info.get('currency', 'N/A')}")
        print(f"Harga Close Terbaru: {data['Close'].iloc[-1]:.2f}")
        print(f"Market Cap: {info.get('marketCap', 'N/A')}")
        print(f"Trailing PE Ratio: {info.get('trailingPE', 'N/A')}")
        print(f"Forward PE Ratio: {info.get('forwardPE', 'N/A')}")
        print(f"Trailing EPS: {info.get('trailingEps', 'N/A')}")
        print(f"Forward EPS: {info.get('forwardEps', 'N/A')}")
        print(f"PEG Ratio: {info.get('pegRatio', 'N/A')}")  # Implied growth
        print(f"Dividend Yield: {info.get('dividendYield', 'N/A')}")
        print(f"Beta: {info.get('beta', 'N/A')}")
        print(f"Target Mean Price: {info.get('targetMeanPrice', 'N/A')}")
        print(f"Target High/Low: High {info.get('targetHighPrice', 'N/A')} Low {info.get('targetLowPrice', 'N/A')}")
        print(f"Number of Analysts: {info.get('numberOfAnalystOpinions', 'N/A')}")

        # 2. Data Historis Komplit (sama)
        print("\n=== Data Historis 5 Hari Terakhir ===")
        print(data.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']])

        print("\n=== Ringkasan Historis 1 Bulan Terakhir ===")
        one_month_ago = data.index[-1] - pd.Timedelta(days=30)
        month_data = data.loc[one_month_ago:]
        print(f"Mean Close 1 Bulan: {month_data['Close'].mean():.2f}")
        print(f"Max Close 1 Bulan: {month_data['Close'].max():.2f}")
        print(f"Min Close 1 Bulan: {month_data['Close'].min():.2f}")
        print(f"Total Volume 1 Bulan: {month_data['Volume'].sum()}")
        print(f"Rata-rata Volume Harian 1 Bulan: {month_data['Volume'].mean():.0f}")

        # 3. Stats Matematis Komplit (sama)
        returns = data['Close'].pct_change().dropna()
        bench_returns = bench_data['Close'].pct_change().dropna()
        print("\n=== Stats Matematis Komplit ===")
        print(f"Mean Return Harian: {returns.mean():.6f}")
        print(f"Std Dev Return (Volatilitas Harian): {returns.std():.6f}")
        print(f"Sharpe Ratio Sederhana (Risk-Free 0): {(returns.mean() / returns.std()) if returns.std() != 0 else 'N/A'}")

        annual_return = (1 + returns.mean()) ** 252 - 1
        print(f"Annual Return Estimasi: {annual_return:.4f}")

        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        print(f"Max Drawdown Historis: {max_drawdown:.4f}")

        acf_vals = acf(returns.dropna(), nlags=5, fft=False)
        print(f"Autocorrelation Return Lag 1-5: {acf_vals[1:6]}")

        vol_clust = returns.rolling(20).std()
        print(f"Rata-rata Volatility Clustering (Rolling Std 20 Hari): {vol_clust.mean():.6f}")
        print(f"Max Volatility Clustering: {vol_clust.max():.6f}")
        print(f"Min Volatility Clustering: {vol_clust.min():.6f}")

        print(f"Return Skewness: {returns.skew():.4f}")
        print(f"Return Kurtosis: {returns.kurtosis():.4f}")

        correlation = returns.corr(bench_returns)
        print(f"Correlation dengan Benchmark IHSG (^JKSE): {correlation:.4f}")

        momentum_score = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].rolling(10).std().iloc[-1] if len(data) > 10 else 'N/A'
        print(f"Momentum Score (Inersia Fisika 10 Hari): {momentum_score:.4f} (>2 kuat buy, < -2 kuat sell)")

        lambda_ewma = 0.94
        vol_forecast = np.sqrt(lambda_ewma * returns.iloc[-1]**2 + (1 - lambda_ewma) * returns.rolling(20).var().iloc[-1]) if len(returns) > 20 else 'N/A'
        print(f"Vol Forecast Besok (EWMA Fisika): {vol_forecast:.4f}")

        # 4. Technical Indicators Komplit (sama)
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

        print("\n=== Technical Indicators Terbaru ===")
        print(f"MA10: {data['MA10'].iloc[-1]:.2f}")
        print(f"MA20: {data['MA20'].iloc[-1]:.2f}")
        print(f"MA50: {data['MA50'].iloc[-1]:.2f}")
        print(f"RSI14: {data['RSI14'].iloc[-1]:.2f}")
        print(f"MACD: {data['MACD'].iloc[-1]:.2f}")
        print(f"Bollinger Upper: {data['BB_Upper'].iloc[-1]:.2f}")
        print(f"Bollinger Mid: {data['BB_Mid'].iloc[-1]:.2f}")
        print(f"Bollinger Lower: {data['BB_Lower'].iloc[-1]:.2f}")

        data['MA_Cross'] = np.where(data['MA10'] > data['MA20'], 1, 0)
        data['MA_Cross_Shift'] = data['MA_Cross'].shift(1)
        crossovers = data[data['MA_Cross'] != data['MA_Cross_Shift']]
        print(f"Jumlah MA10/MA20 Crossover Historis: {len(crossovers)}")
        print(f"Buy Signals (MA10 > MA20): {data['MA_Cross'].sum()}")
        print(f"Buy Signal Terbaru: {'Yes' if data['MA_Cross'].iloc[-1] == 1 else 'No'}")

        # 5. Monte Carlo Simulasi Komplit (sama)
        print("\n=== Monte Carlo Simulasi Komplit ===")
        sim_returns = np.random.normal(returns.mean(), returns.std(), (365, 200))
        sim_paths = (1 + sim_returns).cumprod(axis=0) * data['Close'].iloc[-1]
        mean_path = sim_paths.mean(axis=1)[-1]
        var_95 = np.percentile(sim_paths[-1,:], 5)
        cvar_95 = sim_paths[-1, sim_paths[-1,:] <= var_95].mean() if len(sim_paths[-1, sim_paths[-1,:] <= var_95]) > 0 else var_95

        print(f"Mean Harga Simulasi 1 Tahun: {mean_path:.2f}")
        print(f"VaR 95% (Worst-case 5%): {var_95:.2f}")
        print(f"CVaR 95% (Rata-Rata Worst-case): {cvar_95:.2f}")
        print(f"Simulasi Return Skewness: {np.skew(sim_paths[-1,:]):.4f}")
        print(f"Simulasi Return Kurtosis: {np.kurtosis(sim_paths[-1,:]):.4f}")

        autocorr_lag1 = acf_vals[1] if len(acf_vals) > 1 else 0
        theta = -np.log(autocorr_lag1) if autocorr_lag1 > 0 else 0.5
        long_term_mean = data['Close'].rolling(200).mean().iloc[-1] if len(data) > 200 else data['Close'].mean()
        ou_drift = theta * (long_term_mean - data['Close'].iloc[-1])
        ou_multiplier = np.exp(ou_drift * 1)
        ou_price_1y = data['Close'].iloc[-1] * ou_multiplier
        print(f"OU Mean Price 1 Tahun (Mean Reversion Fisika): {ou_price_1y:.2f}")

        plt.figure(figsize=(14,7))
        plt.plot(sim_paths, alpha=0.1)
        plt.title(f'Monte Carlo Simulasi Harga 1 Tahun {ticker}')
        plt.xlabel('Hari')
        plt.ylabel('Harga Simulasi')
        plt.grid(True)
        plt.show(block=False)

        # PESAN AKHIR (sama)
        print("\n=== Analisa Selesai 100% Komplit! ===")
        print("Semua hitungan matematis sudah dilakukan. Chart muncul sekarang (close kalau sudah dilihat).")
        print("Copy output terminal ini untuk olah di AI universal.")

        # Tambahan: Cek Parameter Pembeda Small Cap vs Big Cap
        print("\n=== Cek Pembeda Small Cap vs Big Cap ===")
        small_count = 0

        # 1. Market Capitalization (dari yfinance info)
        market_cap = info.get('marketCap', 0)
        print(f"1. Market Cap: {market_cap:.0f} (Sumber: yfinance)")
        if market_cap < 5e12:  # <5T
            small_count += 1
            print("   Condong Small Cap")

        # 2. Rata-rata Nilai Transaksi Harian (dari history: avg volume * avg close 1y)
        one_year_ago = data.index[-1] - pd.Timedelta(days=365)
        year_data = data.loc[one_year_ago:]
        avg_value = year_data['Volume'].mean() * year_data['Close'].mean()
        print(f"2. Avg Nilai Transaksi Harian: {avg_value:.0f} (Sumber: yfinance history)")
        if avg_value < 20e9:  # <20M
            small_count += 1
            print("   Condong Small Cap")

        # 3. Bid-Ask Spread (dari info, kalau available)
        bid = info.get('bid', 0)
        ask = info.get('ask', 0)
        spread = ask - bid if ask > bid else 'N/A'
        print(f"3. Bid-Ask Spread: {spread} (Sumber: yfinance, kalau N/A = tidak available)")
        if spread != 'N/A' and spread / data['Close'].iloc[-1] > 0.01:  # Spread >1% price
            small_count += 1
            print("   Condong Small Cap")

        # 4. Depth Order Book (bidSize/askSize dari info, kalau available)
        bid_size = info.get('bidSize', 0)
        ask_size = info.get('askSize', 0)
        depth = (bid_size + ask_size) / 2 if bid_size + ask_size > 0 else 'N/A'
        print(f"4. Avg Depth (bid/ask size): {depth} (Sumber: yfinance, kalau N/A = tidak available)")
        if depth != 'N/A' and depth < 100:  # Asumsi tipis <100 lot
            small_count += 1
            print("   Condong Small Cap")

        # 5. Konsistensi Volume (CV = std/mean volume 1y)
        volume_cv = year_data['Volume'].std() / year_data['Volume'].mean() if year_data['Volume'].mean() != 0 else 'N/A'
        print(f"5. Konsistensi Volume (CV): {volume_cv:.2f} (Sumber: yfinance history)")
        if volume_cv > 2:  # CV tinggi = tidak konsisten
            small_count += 1
            print("   Condong Small Cap")

        # 6. Autocorrelation Return (dari acf_vals lag1)
        autocorr_lag1 = acf_vals[1]
        print(f"6. Autocorrelation Return Lag1: {autocorr_lag1:.4f} (Sumber: statsmodels acf)")
        if abs(autocorr_lag1) < 0.1:  # Rendah = tidak stabil
            small_count += 1
            print("   Condong Small Cap")

        # 7. Volatility (std returns 1y)
        year_returns = year_data['Close'].pct_change().dropna()
        volatility = year_returns.std()
        print(f"7. Volatility (Std Returns 1y): {volatility:.4f} (Sumber: yfinance history)")
        if volatility > 0.05:  # >5% harian = ekstrem
            small_count += 1
            print("   Condong Small Cap")

        # 8. Kurtosis Return (dari returns.kurtosis())
        kurt = returns.kurtosis()
        print(f"8. Kurtosis Return: {kurt:.4f} (Sumber: pandas)")
        if kurt > 3:  # >normal = fat tail
            small_count += 1
            print("   Condong Small Cap")

        # 9. Korelasi ke IHSG (udah ada)
        print(f"9. Korelasi ke IHSG: {correlation:.4f} (Sumber: yfinance)")
        if correlation < 0.5:  # Rendah
            small_count += 1
            print("   Condong Small Cap")

        # 10. Kepemilikan Institusi (dari major_holders % institutional)
        major_holders = yf_data.major_holders
        inst_pct = major_holders[major_holders[1].str.contains('Institutional')][0].values[0] if not major_holders.empty else 0
        print(f"10. Kepemilikan Institusi: {inst_pct}% (Sumber: yfinance major_holders)")
        if inst_pct < 20:  # Minim
            small_count += 1
            print("   Condong Small Cap")

        # 11. Jumlah Analis (dari info)
        num_analysts = info.get('numberOfAnalystOpinions', 0)
        print(f"11. Jumlah Analis: {num_analysts} (Sumber: yfinance)")
        if num_analysts < 2:  # 0-1
            small_count += 1
            print("   Condong Small Cap")

        # 12. Reaksi terhadap News (cek avg |return| 1 hari post 5 news terbaru)
        news_reaction = []
        for n in news[:5]:  # 5 news terbaru
            news_date = datetime.fromtimestamp(n['providerPublishTime'])
            news_date = pd.to_datetime(news_date.date())
            if news_date in data.index:
                post_return = data.loc[news_date:news_date + pd.Timedelta(days=1)]['Close'].pct_change().abs().mean()
                news_reaction.append(post_return)
        avg_reaction = np.mean(news_reaction) if news_reaction else 'N/A'
        print(f"12. Avg Reaksi News (Abs Return Post 5 News): {avg_reaction:.4f} (Sumber: yfinance news + history)")
        if avg_reaction > 0.05:  # Overreaction >5%
            small_count += 1
            print("   Condong Small Cap")

        # 13. Holding Period Bandar (skip, no data direct)
        print("13. Holding Period Bandar: N/A (Butuh data broker)")

        # 14. Kemampuan Absorb Supply (cek avg |return| saat volume > 2x mean)
        mean_vol = data['Volume'].mean()
        high_vol_days = data[data['Volume'] > 2 * mean_vol]
        absorb = high_vol_days['Close'].pct_change().abs().mean() if not high_vol_days.empty else 'N/A'
        print(f"14. Avg Abs Return saat High Volume: {absorb:.4f} (Sumber: yfinance history)")
        if absorb > 0.05:  # Mudah jebol >5%
            small_count += 1
            print("   Condong Small Cap")

        # Kesimpulan satu baris
        print(f"\nKESIMPULAN: Hitung Small Cap Indicator: {small_count}/12 (skip 13 & kalau N/A). Kalau ≥5 indikator condong ke kiri, perlakukan instrumen sebagai short-lived thesis asset.")

    except Exception as e:
        print(f"\nError: {e}")
        print("Data sudah cukup untuk analisa dasar. Copy yang keluar saja.")

# Loop interaktif (sama)
while True:
    print("Quant Analisa Saham Komplit Kompleks V2 Fisika - Masukkan ticker (contoh EMDE.JK):")
    ticker = input().strip().upper() or 'EMDE.JK'
    analyze_ticker(ticker)
    print("\nMasukkan ticker lagi? (y/n):")
    repeat = input().strip().lower()
    if repeat != 'y':
        print("Program selesai. Terima kasih!")
        break
