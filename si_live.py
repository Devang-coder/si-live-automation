# si_live.py — Automated SI Live update (downloads from Google Drive, Prophet-free)

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

warnings.filterwarnings("ignore")
print("✅ Libraries imported successfully.")

# === 1) Google Drive auth (service account is written by GitHub Actions as service_account.json)
SERVICE_ACCOUNT_FILE = "service_account.json"
SCOPES = ["https://www.googleapis.com/auth/drive"]
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive = build("drive", "v3", credentials=creds)

# === 2) Google Drive file IDs
# You gave these links:
#   xgb_model_columns.json -> https://drive.google.com/file/d/1QfLLd318OnSafRJZNVubmMgVQqfKib5N/view
#   xgb_portfolio_model.pkl -> https://drive.google.com/file/d/1b37nDV7ZZl3pEZM6xDlIDWqNkToKc0DA/view
COLUMNS_FILE_ID = "1QfLLd318OnSafRJZNVubmMgVQqfKib5N"
MODEL_FILE_ID = "1b37nDV7ZZl3pEZM6xDlIDWqNkToKc0DA"

# If you know the file ID of "Devang Portfolio 25.xlsx", set it here.
# If left as None, we will search Drive by name and pick the first match.
PORTFOLIO_FILE_ID = None
PORTFOLIO_FILENAME = "Devang Portfolio 25.xlsx"

# This is the Drive file ID of live_portfolio_results.json (the one we keep updating)
RESULT_JSON_FILE_ID = "1FtxoyquFx3q0wU1QELX830UfrByvIRou"

# === 3) Local paths (GitHub runner workspace)
COLUMNS_PATH = "xgb_model_columns.json"
MODEL_PATH = "xgb_portfolio_model.pkl"
PORTFOLIO_PATH = "Devang Portfolio 25.xlsx"
OUTPUT_PATH = "live_portfolio_results.json"


# === Helpers ===
def download_file(file_id: str, dest_path: str) -> None:
    """Download a file from Drive to local path."""
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"⬇️  Downloading {dest_path}: {int(status.progress() * 100)}%")
    print(f"✅ Downloaded: {dest_path}")


def find_drive_file_id_by_name(name: str) -> str | None:
    """Find the first non-trashed file with the given name and return its ID."""
    response = (
        drive.files()
        .list(
            q=f"name = '{name}' and trashed = false",
            fields="files(id, name)",
            pageSize=5,
            spaces="drive",
        )
        .execute()
    )
    files = response.get("files", [])
    return files[0]["id"] if files else None


# === 4) Download required artifacts from Drive ===
print("🔎 Preparing artifacts from Google Drive...")
# Columns JSON
download_file(COLUMNS_FILE_ID, COLUMNS_PATH)

# Model PKL
download_file(MODEL_FILE_ID, MODEL_PATH)

# Portfolio Excel
portfolio_id = PORTFOLIO_FILE_ID or find_drive_file_id_by_name(PORTFOLIO_FILENAME)
if not portfolio_id:
    raise RuntimeError(
        f"Could not find '{PORTFOLIO_FILENAME}' in Drive. "
        f"Please set PORTFOLIO_FILE_ID explicitly."
    )
download_file(portfolio_id, PORTFOLIO_PATH)


# === 5) Load model + columns ===
print("🧠 Loading model + feature columns...")
try:
    model = joblib.load(MODEL_PATH)
    with open(COLUMNS_PATH, "r") as f:
        model_columns = json.load(f)
    print("✅ Model and columns loaded.")
except Exception as e:
    raise RuntimeError(f"Failed to load model or columns: {e}") from e


# === 6) Load latest portfolio snapshot from Excel ===
print("📄 Reading portfolio snapshot from Excel...")
full_sheet_df = pd.read_excel(PORTFOLIO_PATH, header=None)
header_indices = full_sheet_df[full_sheet_df[0] == "Top Stock"].index
if len(header_indices) == 0:
    raise RuntimeError("Could not find 'Top Stock' header in the Excel file.")

last_header_idx = header_indices[-1]
headers = full_sheet_df.iloc[last_header_idx].tolist()
last_snapshot_df = full_sheet_df.iloc[last_header_idx + 1 : last_header_idx + 11].copy()
last_snapshot_df.columns = headers
last_snapshot_df = last_snapshot_df.reset_index(drop=True)
print("✅ Portfolio snapshot loaded.")


# === 7) Get live prices via yfinance (2y history) ===
stock_names = last_snapshot_df["Top Stock"].unique()
yf_tickers = [f"{s}.NS" for s in stock_names]
print(f"📈 Fetching 2y history for: {yf_tickers}")

history_data = yf.download(yf_tickers, period="2y")["Close"]
if history_data is None or history_data.empty:
    print("⚠️ yfinance returned no data. Falling back to Excel LTP values.")
    live_price_map = pd.Series(
        last_snapshot_df["LTP"].values, index=last_snapshot_df["Top Stock"]
    ).to_dict()
    history_data = None
else:
    last_prices = history_data.iloc[-1]
    live_price_map = {t.split(".")[0]: last_prices[t] for t in yf_tickers}
    print("✅ Live prices fetched.")


# === 8) Build live snapshot & features ===
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

# One-hot for model
live_df_encoded = pd.get_dummies(live_df, columns=["Top Stock"], drop_first=True)
live_X = pd.DataFrame(columns=model_columns)
live_X = pd.concat([live_X, live_df_encoded]).fillna(0)[model_columns]

# === 9) XGB prediction ===
predicted_prices_xgb = model.predict(live_X)

# === 10) Monte Carlo + ARIMA (skip if no history) ===
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
        return float("nan")


if history_data is not None:
    monte_carlo_results, arima_results = [], []
    for s in stock_names:
        series = history_data[f"{s}.NS"].dropna()
        monte_carlo_results.append(run_monte_carlo(series))
        arima_results.append(run_arima_forecast(series))
else:
    monte_carlo_results = [float("nan")] * len(stock_names)
    arima_results = [float("nan")] * len(stock_names)

print("✅ Monte Carlo + ARIMA complete.")

# === 11) Build results + save JSON ===
results_df = pd.DataFrame(
    {
        "Stock": last_snapshot_df["Top Stock"],
        "Current_LTP": live_df["LTP"].astype(float),
        "XGB_Prediction": predicted_prices_xgb.astype(float),
        "ARIMA_90d_Forecast": arima_results,
        "VaR_5_Pct_90d": monte_carlo_results,
    }
)
results_df["XGB_Change_%"] = (
    (results_df["XGB_Prediction"] - results_df["Current_LTP"])
    / results_df["Current_LTP"]
    * 100
)
results_df["ARIMA_Change_%"] = (
    (results_df["ARIMA_90d_Forecast"] - results_df["Current_LTP"])
    / results_df["Current_LTP"]
    * 100
)

output = {
    "last_updated_iso": datetime.now().isoformat(),
    "status": "Success",
    "data": results_df.to_dict("records"),
}
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=4)
print(f"✅ JSON saved to {OUTPUT_PATH}")

# === 12) Upload JSON back to Drive ===
try:
    media = MediaFileUpload(OUTPUT_PATH, mimetype="application/json", resumable=True)
    drive.files().update(fileId=RESULT_JSON_FILE_ID, media_body=media).execute()
    print("✅ Uploaded updated JSON to Google Drive.")
except Exception as e:
    print(f"❌ Failed to upload JSON to Drive: {e}")
