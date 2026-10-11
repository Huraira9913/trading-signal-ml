"""
Simple, reproducible trading signal generator using synthetic OHLCV data.
Demonstrates feature engineering + ensemble voting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def generate_ohlcv(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="h")
    returns = rng.normal(0.0001, 0.01, n)
    price = 100 * np.cumprod(1 + returns)
    high = price * (1 + rng.uniform(0, 0.005, n))
    low = price * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(1000, 10000, n)
    return pd.DataFrame({
        "timestamp": dates,
        "open": price,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    df["volatility_20"] = df["ret_1"].rolling(20).std()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["distance_sma"] = (df["close"] - df["sma_50"]) / df["sma_50"]
    df["volume_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    # Target: next 5-period return > 0
    df["target"] = (df["close"].shift(-5) > df["close"]).astype(int)
    return df.dropna()


def train_signal_model(df: pd.DataFrame):
    features = ["ret_1", "ret_5", "ret_20", "volatility_20", "distance_sma", "volume_z"]
    X = df[features]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("=== Trading Signal Model ===")
    print(classification_report(y_test, y_pred, target_names=["Down/Flat", "Up"]))
    print("Feature importances:")
    for f, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f}: {imp:.4f}")
    
    return model


if __name__ == "__main__":
    df = generate_ohlcv()
    feat_df = engineer_features(df)
    print(f"Feature matrix shape: {feat_df.shape}")
    train_signal_model(feat_df)
