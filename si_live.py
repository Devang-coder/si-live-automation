# si_live.py — Automated SI Live update (Prophet-free, GitHub-ready)

import pandas as pd
import yfinance as yf
import joblib
import json
import numpy as np
import warnings
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import os

warnings.filterwarnings('ignore')
print("✅ Libraries imported successfully.")

# === 1️⃣ Google Drive Setup ===
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Created by GitHub Actions from your secret
SCOPES = ['https://www.googleapis.com/auth/drive']
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

# === 2️⃣ File Paths (local) ===
MODEL_PATH = 'xgb_portfolio_model.pkl'
COLUMNS_PATH = 'xgb_model_columns.json'
PORTFOLIO_PATH = 'Devang Portfolio 25.xlsx'
OUTPUT_PATH = 'live_portfolio_results.json'

# === 3️⃣ Load Model and Columns ===
print("Loading model and feature columns...")
try:
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH, 'r') as f:
        model_columns = json.load(f)
    print("✅ Model and columns loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# === 4️⃣ Load Excel Portfolio Snapshot ===
try:
    full_sheet_df = pd.read_excel(PORTFOLIO_PATH, header=None)
    header_indices = full_sheet_df[full_sheet_df[0] == 'Top Stock'].index
    last_header_idx = header_indices[-1]
    headers = full_sheet_df.iloc[last_header_idx].tolist()
    last_snapshot_df = full_sheet_df.iloc[last_header_idx + 1:last_header_idx + 11].copy()
    last_snapshot_df.columns = headers
    last_snapshot_df = last_snapshot_df.reset_index(drop=True)
    print("✅ Portfolio snapshot loaded.")
except Exception as e:
    print(f"❌ Error loading Excel file: {e}")
    exit(1)

# === 5️⃣ Fetch 2-Year Price History ===
stock_names = last_snapshot_df['Top Stock'].unique()
yf_tickers = [f"{s}.NS" for s in stock_names]
print(f"📈 Fetching 2 years of price history for: {yf_tickers}")

try:
    history_data = yf.download(yf_tickers, period="2y")['Close']
    last_prices = history_data.iloc[-1]
    live_price_map = {ticker.split('.')[0]: last_prices[ticker] for ticker in yf_tickers}
    print("✅ Historical data fetched successfully.")
except Exception as e:
    print(f"⚠️ yfinance fetch failed: {e}")
    live_price_map = pd.Series(
        last_snapshot_df.LTP.values,
        index=last_snapshot_df['Top Stock']
    ).to_dict()
    history_data = None

# === 6️⃣ Build Live Snapshot ===
live_df = last_snapshot_df.copy()
live_df['Prev_LTP'] = live_df['LTP']
live_df['LTP'] = live_df['Top Stock'].map(live_price_map)
numeric_cols = ['Qty', 'Buy Price', 'MBP', 'LTP', 'Prev_LTP']
live_df[numeric_cols] = live_df[numeric_cols].astype(float)
live_df['%Change'] = (live_df['LTP'] - live_df['Prev_LTP']) / live_df['Prev_LTP'] * 100
live_df['LTP_Buy_diff'] = live_df['LTP'] - live_df['Buy Price']
live_df['MBP_LTP_diff'] = live_df['LTP'] - live_df['MBP']
live_df['ReturnPct'] = (live_df['LTP'] - live_df['Buy Price']) / live_df['Buy Price'] * 100
print("✅ Live snapshot prepared.")

# === 7️⃣ Prepare Features for Model ===
live_df_encoded = pd.get_dummies(live_df, columns=['Top Stock'], drop_first=True)
live_X = pd.DataFrame(columns=model_columns)
live_X = pd.concat([live_X, live_df_encoded]).fillna(0)[model_columns]

# === 8️⃣ Make Predictions ===
predicted_prices_xgb = model.predict(live_X)

# === 9️⃣ Monte Carlo + ARIMA Forecasts ===
def run_monte_carlo(price_history, horizon_days=90, n_sims=1000):
    prices = price_history.dropna().values
    if len(prices) < 30:
        return np.nan
    log_returns = np.diff(np.log(prices))
    mu, sigma = log_returns.mean(), log_returns.std()
    sims = np.zeros((n_sims, horizon_days))
    last_price = prices[-1]
    for i in range(n_sims):
        rand_returns = np.random.normal(mu, sigma, horizon_days)
        sims[i] = last_price * np.exp(np.cumsum(rand_returns))
    final_prices = sims[:, -1]
    returns_pct = (final_prices - last_price) / last_price * 100
    return np.percentile(returns_pct, 5)

def run_arima_forecast(price_series, horizon_days=90):
    try:
        series = price_series.astype(float).values
        model = ARIMA(series, order=(5, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=horizon_days)
        return forecast[-1]
    except Exception:
        return np.nan

monte_carlo_results, arima_results = [], []
if history_data is not None:
    for stock in stock_names:
        ticker = f"{stock}.NS"
        price_series = history_data[ticker].dropna()
        monte_carlo_results.append(run_monte_carlo(price_series))
        arima_results.append(run_arima_forecast(price_series))
else:
    monte_carlo_results = [np.nan] * len(stock_names)
    arima_results = [np.nan] * len(stock_names)

print("✅ Monte Carlo and ARIMA forecasts complete.")

# === 🔟 Combine All Results ===
results_df = pd.DataFrame({
    'Stock': last_snapshot_df['Top Stock'],
    'Current_LTP': live_df['LTP'],
    'XGB_Prediction': predicted_prices_xgb,
    'ARIMA_90d_Forecast': arima_results,
    'VaR_5_Pct_90d': monte_carlo_results
})
results_df['XGB_Change_%'] = (
    (results_df['XGB_Prediction'] - results_df['Current_LTP'])
    / results_df['Current_LTP'] * 100
)
results_df['ARIMA_Change_%'] = (
    (results_df['ARIMA_90d_Forecast'] - results_df['Current_LTP'])
    / results_df['Current_LTP'] * 100
)

# === 1️⃣1️⃣ Save JSON Output Locally ===
output = {
    'last_updated_iso': datetime.now().isoformat(),
    'status': 'Success',
    'data': results_df.to_dict('records')
}
with open(OUTPUT_PATH, 'w') as f:
    json.dump(output, f, indent=4)
print(f"✅ JSON saved locally to {OUTPUT_PATH}")

# === 1️⃣2️⃣ Upload Updated JSON to Google Drive ===
# Your exact Drive file ID (from your link)
DRIVE_FILE_ID = "1FtxoyquFx3q0wU1QELX830UfrByvIRou"

try:
    media = MediaFileUpload(OUTPUT_PATH, mimetype='application/json', resumable=True)
    drive_service.files().update(fileId=DRIVE_FILE_ID, media_body=media).execute()
    print("✅ Uploaded updated JSON to Google Drive successfully.")
except Exception as e:
    print(f"❌ Failed to upload to Google Drive: {e}")
