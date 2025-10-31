# si_live.py — Automated SI Live update (downloads from Google Drive, now includes Prophet forecast)
import io
import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet  # ✅ Added Prophet

warnings.filterwarnings("ignore")
print("✅ Libraries imported successfully.")

# === 1️⃣ Google Drive auth (service account from GitHub Secrets) ===
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive = build("drive", "v3", credentials=creds)

# === 2️⃣ Google Drive file IDs ===
COLUMNS_FILE_ID = "1QfLLd318OnSafRJZNVubmMgVQqfKib5N"
MODEL_FILE_ID = "1b37nDV7ZZl3pEZM6xDlIDWqNkToKc0DA"
PORTFOLIO_FILE_ID = "1GF4NQi92ojcpgUoikO4zIltfEkgiZuBy"
RESULT_JSON_FILE_ID = "1FtxoyquFx3q0wU1QELX830UfrByvIRou"

# === 3️⃣ Local file paths ===
COLUMNS_PATH = "xgb_model_columns.json"
MODEL_PATH = "xgb_portfolio_model.pkl"
PORTFOLIO_PATH = "Devang_Portfolio_25.xlsx"
OUTPUT_PATH = "live_portfolio_results.json"


# === 4️⃣ Helper functions ===
def download_file(file_id: str, dest_path: str) -> None:
    """Download a file from Google Drive."""
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"⬇️  Downloading {dest_path}: {int(status.progress() * 100)}%")
    print(f"✅ Downloaded: {dest_path}")


def download_google_sheet_as_excel(sheet_id: str, dest_path: str):
    """Download a Google Sheet as .xlsx"""
    print("📄 Downloading Google Sheet as Excel...")
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    from urllib.request import urlopen

    try:
        response = urlopen(export_url)
        with open(dest_path, "wb") as f:
            f.write(response.read())
        print(f"✅ Google Sheet saved as: {dest_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to download Google Sheet: {e}")


# === 5️⃣ Download model, columns, and portfolio ===
print("🔎 Preparing artifacts from Google Drive...")
download_file(COLUMNS_FILE_ID, COLUMNS_PATH)
download_file(MODEL_FILE_ID, MODEL_PATH)
download_google_sheet_as_excel(PORTFOLIO_FILE_ID, PORTFOLIO_PATH)

# === 6️⃣ Load model and feature columns ===
print("🧠 Loading model and columns...")
model = joblib.load(MODEL_PATH)
with open(COLUMNS_PATH, "r") as f:
    model_columns = json.load(f)
print("✅ Model and columns loaded.")


# === 7️⃣ Read latest portfolio snapshot ===
full_sheet_df = pd.read_excel(PORTFOLIO_PATH, header=None)
header_indices = full_sheet_df[full_sheet_df[0] == "Top Stock"].index
if len(header_indices) == 0:
    raise RuntimeError("❌ Could not find 'Top Stock' header in the Excel file.")

last_header_idx = header_indices[-1]
headers = full_sheet_df.iloc[last_header_idx].tolist()
last_snapshot_df = full_sheet_df.iloc[last_header_idx + 1 : last_header_idx + 11].copy()
last_snapshot_df.columns = headers
last_snapshot_df = last_snapshot_df.reset_index(drop=True)
print("✅ Portfolio snapshot loaded.")


# === 8️⃣ Fetch live prices from Yahoo Finance ===
stock_names = last_snapshot_df["Top Stock"].unique()
yf_tickers = [f"{s}.NS" for s in stock_names]
print(f"📈 Fetching 2-year price history for: {yf_tickers}")

try:
    history_data = yf.download(yf_tickers, period="2y")["Close"]
    last_prices = history_data.iloc[-1]
    live_price_map = {ticker.split(".")[0]: last_prices[ticker] for ticker in yf_tickers}
    print("✅ Live prices fetched successfully.")
except Exception as e:
    print(f"⚠️ yfinance failed ({e}), falling back to Excel values.")
    live_price_map = pd.Series(last_snapshot_df["LTP"].values, index=last_snapshot_df["Top Stock"]).to_dict()
    history_data = None


# === 9️⃣ Prepare live snapshot ===
live_df = last_snapshot_df.copy()
live_df["Prev_LTP"] = live_df["LTP"]
live_df["LTP"] = live_df["Top Stock"].map(live_price_map)

numeric_cols = ["Qty", "Buy Price", "MBP", "LTP", "Prev_LTP"]
live_df[numeric_cols] = live_df[numeric_cols].astype(float)
live_df["%Change"] = (live_df["LTP"] - live_df["Prev_LTP"]) / live_df["Prev_LTP"] * 100
live_df["LTP_Buy_diff"] = live_df["LTP"] - live_df["Buy Price"]
live_df["MBP_LTP_diff"] = live_df["LTP"] - live_df["MBP"]
live_df["ReturnPct"] = (live_df["LTP"] - live_df["Buy Price"]) / live_df["Buy Price"] * 100
print("✅ Live snapshot prepared.")


# === 🔟 Feature engineering for ML ===
live_df_encoded = pd.get_dummies(live_df, columns=["Top Stock"], drop_first=True)
live_X = pd.DataFrame(columns=model_columns)
live_X = pd.concat([live_X, live_df_encoded]).fillna(0)[model_columns]

# === 1️⃣1️⃣ Model prediction ===
predicted_prices_xgb = model.predict(live_X)


# === 1️⃣2️⃣ Forecasting (Prophet + ARIMA + Monte Carlo) ===
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
    return float(np.percentile(returns_pct, 5))


def run_arima_forecast(price_series, horizon_days=90):
    try:
        series = price_series.astype(float).values
        model_ = ARIMA(series, order=(5, 1, 2))
        model_fit = model_.fit()
        forecast = model_fit.forecast(steps=horizon_days)
        return float(forecast[-1])
    except Exception:
        return np.nan


def run_prophet_forecast(price_series, horizon_days=90):
    try:
        df = pd.DataFrame({"ds": price_series.index, "y": price_series.values})
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)
        return float(forecast["yhat"].iloc[-1])
    except Exception:
        return np.nan


prophet_results, arima_results, monte_carlo_results = [], [], []

if history_data is not None:
    print("🔮 Running Prophet + ARIMA + Monte Carlo...")
    for stock in stock_names:
        series = history_data[f"{stock}.NS"].dropna()
        prophet_results.append(run_prophet_forecast(series))
        arima_results.append(run_arima_forecast(series))
        monte_carlo_results.append(run_monte_carlo(series))
else:
    prophet_results = [np.nan] * len(stock_names)
    arima_results = [np.nan] * len(stock_names)
    monte_carlo_results = [np.nan] * len(stock_names)

print("✅ Forecasting complete.")


# === 1️⃣3️⃣ Combine results and save JSON ===
results_df = pd.DataFrame({
    "Stock": last_snapshot_df["Top Stock"],
    "Current_LTP": live_df["LTP"],
    "XGB_Prediction": predicted_prices_xgb,
    "Prophet_90d_Forecast": prophet_results,
    "ARIMA_90d_Forecast": arima_results,
    "VaR_5_Pct_90d": monte_carlo_results,
})
results_df["XGB_Change_%"] = (results_df["XGB_Prediction"] - results_df["Current_LTP"]) / results_df["Current_LTP"] * 100
results_df["Prophet_Change_%"] = (results_df["Prophet_90d_Forecast"] - results_df["Current_LTP"]) / results_df["Current_LTP"] * 100
results_df["ARIMA_Change_%"] = (results_df["ARIMA_90d_Forecast"] - results_df["Current_LTP"]) / results_df["Current_LTP"] * 100

output = {
    "last_updated_iso": datetime.now().isoformat(),
    "status": "Success",
    "data": results_df.to_dict("records"),
}

# ✅ FIXED: Replace NaN, inf, -inf with None before saving to JSON
def clean_nans(obj):
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        else:
            return obj
    else:
        return obj

cleaned_output = clean_nans(output)

with open(OUTPUT_PATH, "w") as f:
    json.dump(cleaned_output, f, indent=4, ensure_ascii=False)

print(f"✅ JSON saved locally to {OUTPUT_PATH}")


# === 1️⃣4️⃣ Upload result JSON back to Google Drive ===
try:
    # Verify JSON is valid before upload
    with open(OUTPUT_PATH, "r") as f:
        json.load(f)
    media = MediaFileUpload(OUTPUT_PATH, mimetype="application/json", resumable=True)
    drive.files().update(fileId=RESULT_JSON_FILE_ID, media_body=media).execute()
    print("✅ Uploaded updated JSON to Google Drive successfully.")
except json.JSONDecodeError:
    print("❌ JSON invalid — skipping upload.")
except Exception as e:
    print(f"❌ Failed to upload to Drive: {e}")
