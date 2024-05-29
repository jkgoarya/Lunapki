import pandas as pd
import numpy as np
import joblib
import logging
import time
from binance.client import Client
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Set up logging
logging.basicConfig(filename='trading_bot.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

# Binance API credentials (replace with your own)
api_key = 
api_secret =

# Initialize Binance client
client = Client(api_key, api_secret)

# Load the trained model and scaler
model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

# Fetch latest data from Binance
def fetch_latest_data(symbol='ETHUSDT', interval='1m'):
    try:
        klines = client.get_historical_klines(symbol, interval, '1 minute ago UTC')
        if klines:
            data = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            logging.info(f"Fetched data: {data.tail(1)}")
            return data.tail(1)
        else:
            logging.error(f"No data fetched for {symbol} at interval {interval}")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Calculate features
def calculate_features(data):
    try:
        data['Price_Change'] = (data['Close'] - data['Open']) / data['Open']
        data['Volatility'] = (data['High'] - data['Low']) / data['Open']
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['EMA_5'] = data['Close'].ewm(span=5).mean()
        data['EMA_10'] = data['Close'].ewm(span=10).mean()

        def compute_rsi(data, window=14):
            delta = data.diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        data['RSI_14'] = compute_rsi(data['Close'])

        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']

        data['BB_upper'] = data['SMA_10'] + (data['Close'].rolling(window=10).std() * 2)
        data['BB_lower'] = data['SMA_10'] - (data['Close'].rolling(window=10).std() * 2)

        data.dropna(inplace=True)
        return data
    except Exception as e:
        logging.error(f"Error calculating features: {e}")
        return pd.DataFrame()

# Function to make trading decision
def make_trade_decision(data):
    try:
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Volatility', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI_14', 'MACD', 'BB_upper', 'BB_lower']
        X = data[features]
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        return prediction[0]
    except Exception as e:
        logging.error(f"Error making trade decision: {e}")
        return None

# Main function
def main():
    while True:
        latest_data = fetch_latest_data()
        if not latest_data.empty:
            features_data = calculate_features(latest_data)
            if not features_data.empty:
                decision = make_trade_decision(features_data)
                if decision is not None:
                    logging.info(f"Trade Decision: {'Buy' if decision == 1 else 'Sell'}")
                    print(f"Trade Decision: {'Buy' if decision == 1 else 'Sell'}")
                    # Here you would add code to execute the trade using Binance API
                    # e.g., client.order_market_buy(symbol='ETHUSDT', quantity=0.01) for Buy
                    # and client.order_market_sell(symbol='ETHUSDT', quantity=0.01) for Sell

        time.sleep(60)  # Wait for 1 minute before fetching new data

# Visualize the data in 3D
def visualize_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data['Open'], data['Close'], data['Volume'], c='r', marker='o')
    ax.set_xlabel('Open Price')
    ax.set_ylabel('Close Price')
    ax.set_zlabel('Volume')
    plt.show()

# Load preprocessed data for visualization
def load_preprocessed_data(file_path):
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Loaded preprocessed data from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error reading preprocessed data from {file_path}: {e}")
            return pd.DataFrame()
    else:
        logging.error(f"File {file_path} does not exist or is empty")
        return pd.DataFrame()

preprocessed_data = load_preprocessed_data('preprocessed_data.csv')
if not preprocessed_data.empty:
    visualize_3d(preprocessed_data)

if __name__ == "__main__":
    main()
