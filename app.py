from flask import Flask, jsonify
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import ta
import os

app = Flask(__name__)

print("=== Flask app starting... ===")

# 모델 및 스케일러 자동 다운로드 (GitHub 사용)
def download_file_with_requests(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename} via requests...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename} ({os.path.getsize(filename)} bytes)")
    else:
        print(f"{filename} already exists ({os.path.getsize(filename)} bytes)")

# GitHub Raw 파일 링크
H5_URL = "https://raw.githubusercontent.com/Rrrrrrasd/predictor-assets/main/predictor-assets/lstm_btc_model10.h5"
SCALE_URL = "https://raw.githubusercontent.com/Rrrrrrasd/predictor-assets/main/predictor-assets/scaler_btc_model10.save"
CSV_URL = "https://raw.githubusercontent.com/Rrrrrrasd/predictor-assets/main/predictor-assets/Bitcoin_Pulse_Hourly_Dataset_from_Markets_Trends_and_Fear.csv"

download_file_with_requests("lstm_btc_model10.h5", H5_URL)
download_file_with_requests("scaler_btc_model10.save", SCALE_URL)
download_file_with_requests("Bitcoin_Pulse_Hourly_Dataset_from_Markets_Trends_and_Fear.csv", CSV_URL)

print("=== Download complete. Loading model and scaler... ===")

# 모델 및 스케일러 로딩
model = load_model("lstm_btc_model10.h5")
scaler = joblib.load("scaler_btc_model10.save")

# 온체인 데이터 로드
pulse_data = pd.read_csv("Bitcoin_Pulse_Hourly_Dataset_from_Markets_Trends_and_Fear.csv")
pulse_data['timestamp'] = pd.to_datetime(pulse_data['timestamp'])
pulse_data_daily = pulse_data.resample('1D', on='timestamp').mean()

FEATURES = [
    'Close_BTC-USD', 'High_BTC-USD', 'Low_BTC-USD', 'Open_BTC-USD', 'Volume_BTC-USD',
    'MA5', 'MA10', 'MA15', 'MA20', 'days_since_halving',
    'btc_dominance', 'altcoin_market_cap', 'fear_greed_index', 'trend_bitcoin', 'trend_buy_crypto',
    'btc_dominance_missing', 'altcoin_market_cap_missing', 'fear_greed_index_missing',
    'trend_bitcoin_missing', 'trend_buy_crypto_missing', 'price_change', 'volatility',
    'price_position', 'volume_spike', 'days_since_peak', 'return_7d',
    'rsi', 'macd', 'ma_ratio_5_20'
]

def fetch_btc_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30&interval=daily"
    response = requests.get(url)

    # 응답 오류 처리
    if response.status_code != 200:
        raise ValueError(f"API 요청 실패: {response.status_code} - {response.text[:200]}")

    try:
        data = response.json()
    except Exception as e:
        raise ValueError(f"응답 JSON 파싱 실패: {e} - 원문: {response.text[:200]}")

    prices = data.get("prices")
    if not prices:
        raise ValueError("API 응답에 'prices' 항목 없음")

    df = pd.DataFrame(prices, columns=['timestamp', 'Close_BTC-USD'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df['Open_BTC-USD'] = df['Close_BTC-USD']
    df['High_BTC-USD'] = df['Close_BTC-USD']
    df['Low_BTC-USD'] = df['Close_BTC-USD']
    df['Volume_BTC-USD'] = 1.0
    return df


def add_features(df):
    df['MA5'] = df['Close_BTC-USD'].rolling(window=5).mean()
    df['MA10'] = df['Close_BTC-USD'].rolling(window=10).mean()
    df['MA15'] = df['Close_BTC-USD'].rolling(window=15).mean()
    df['MA20'] = df['Close_BTC-USD'].rolling(window=20).mean()
    df['price_change'] = df['Close_BTC-USD'].pct_change().fillna(0)
    df['volatility'] = df['Close_BTC-USD'].rolling(window=20).std().fillna(0)
    df['rsi'] = ta.momentum.RSIIndicator(df['Close_BTC-USD']).rsi()
    df['macd'] = ta.trend.MACD(df['Close_BTC-USD']).macd_diff()
    df['ma_ratio_5_20'] = df['MA5'] / df['MA20']

    rolling_mean = df['Close_BTC-USD'].rolling(window=20).mean()
    rolling_std = df['Close_BTC-USD'].rolling(window=20).std()
    boll_upper = rolling_mean + 2 * rolling_std
    boll_lower = rolling_mean - 2 * rolling_std
    df['price_position'] = (df['Close_BTC-USD'] - boll_lower) / (boll_upper - boll_lower)

    volume_mean = df['Volume_BTC-USD'].rolling(window=20).mean()
    df['volume_spike'] = df['Volume_BTC-USD'] / volume_mean
    peak_idx = df['Close_BTC-USD'].expanding().apply(lambda x: x.argmax())
    df['days_since_peak'] = df.index.to_series().reset_index(drop=True).index - peak_idx.astype(int)
    df['return_7d'] = df['Close_BTC-USD'].pct_change(periods=7).fillna(0)

    halving_dates = pd.to_datetime(['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'])
    df['days_since_halving'] = df.index.map(lambda x: min([(x - d).days for d in halving_dates if x > d] or [0]))

    for col in ['MA5', 'MA10', 'MA15', 'MA20', 'price_change', 'volatility', 'rsi', 'macd',
                'ma_ratio_5_20', 'price_position', 'volume_spike', 'days_since_peak', 'return_7d']:
        df[col] = df[col].fillna(0)
    return df

def merge_onchain(df):
    joined = df.copy()
    onchain = pulse_data_daily.reindex(joined.index)
    for col in onchain.columns:
        if not col.endswith("_missing"):
            onchain[col] = onchain[col].fillna(-1)
            onchain[f"{col}_missing"] = (onchain[col] == -1).astype(int)
    return joined.join(onchain)

@app.route("/predict", methods=["GET"])
def predict():
    try:
        df = fetch_btc_data()
        df = add_features(df)
        df = merge_onchain(df)
        df = df.dropna(subset=['MA20', 'Close_BTC-USD'])

        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[FEATURES]

        if len(df) < 30:
            return jsonify({"error": "not enough data"}), 400

        input_seq_df = df[-30:][FEATURES]
        input_scaled = scaler.transform(input_seq_df).reshape(1, 30, len(FEATURES))
        pred_scaled = model.predict(input_scaled)[0][0]

        dummy_input = [pred_scaled] + [0] * (scaler.n_features_in_ - 1)
        dummy_df = pd.DataFrame([dummy_input], columns=scaler.feature_names_in_)
        pred_price = scaler.inverse_transform(dummy_df)[0][0]

        return jsonify({"predicted_close": round(pred_price, 2)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"=== Running Flask on port {port} ===")
    app.run(host="0.0.0.0", port=port)