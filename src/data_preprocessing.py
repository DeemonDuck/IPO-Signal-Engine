import pandas as pd
import numpy as np
import yfinance as yf
import os
import yaml


# -----------------------------
# 0. Load Config
# -----------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# -----------------------------
# 1. Download Nifty Data
# -----------------------------
def fetch_nifty_data(start, end):
    nifty = yf.download("^NSEI", start=start, end=end)

    # Flatten columns if multi-index
    nifty.columns = nifty.columns.get_level_values(0)

    nifty = nifty.reset_index()[["Date", "Close"]]

    # Compute 7-day return
    nifty["Nifty_Return_7d"] = nifty["Close"].pct_change(7) * 100

    return nifty[["Date", "Nifty_Return_7d"]]


# -----------------------------
# 2. Load IPO Data
# -----------------------------
def load_ipo_data(file_path):
    df = pd.read_excel(file_path)
    return df


# -----------------------------
# 3. Feature Engineering
# -----------------------------
def engineer_features(df, nifty_features):

    df = df.copy()

    # Date processing
    df["Date"] = pd.to_datetime(df["Date"])
    nifty_features["Date"] = pd.to_datetime(nifty_features["Date"])

    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    # Feature engineering
    df["Demand_Gap"] = df["QIB"] - df["RII"]
    df["log_issue_size"] = np.log1p(df["Issue_Size(crores)"])  # safer log

    # Target variable
    df["Apply_Label"] = (df["Listing Gain"] > 10).astype(int)

    # Merge Nifty data
    df = pd.merge(df, nifty_features, on="Date", how="left")

    # Fill missing values (IPO dates may not align with market days)
    df["Nifty_Return_7d"] = df["Nifty_Return_7d"].ffill()

    return df


# -----------------------------
# 4. Remove Leakage Columns
# -----------------------------
def remove_leakage(df):
    cols_to_drop = [
        "IPO_Name",
        "List Price",
        "CMP(BSE)",
        "CMP(NSE)",
        "Current Gains",
        "Listing Gain",
        "Date"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


# -----------------------------
# 5. Final Cleaning
# -----------------------------
def clean_dataset(df):

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # -----------------------------
    # EDA-based cleaning (IMPORTANT)
    # -----------------------------

    # Drop rows with missing values
    df = df.dropna()

    # Drop redundant column identified during EDA
    df = df.drop(columns=["Total"], errors="ignore")

    return df

# -----------------------------
# 6. Save Dataset
# -----------------------------
def save_dataset(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved at: {output_path}")


# -----------------------------
# 7. Main Pipeline
# -----------------------------
def run_pipeline(config):
    print("Starting data pipeline...")

    input_path = config["data"]["raw_path"]
    output_path = config["data"]["processed_path"]

    start_date = config["nifty"]["start_date"]
    end_date = config["nifty"]["end_date"]

    nifty_features = fetch_nifty_data(start=start_date, end=end_date)
    ipo_df = load_ipo_data(input_path)

    df = engineer_features(ipo_df, nifty_features)
    df = remove_leakage(df)
    df = clean_dataset(df)

    save_dataset(df, output_path)

    print("\nFinal Shape:", df.shape)
    print(df.head())


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)