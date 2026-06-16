# ============================================================
#  MacroSentinel v3 — Flask Backend
#  Loads the trained LSTM model and serves real predictions
#  via a REST API that the frontend calls.
#
#  Endpoints:
#    GET  /              → serves the dashboard HTML
#    GET  /api/predict   → returns real LSTM predictions + metrics
#    GET  /api/prices    → returns latest EUR/USD & GBP/USD prices
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             precision_score, recall_score, f1_score)
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)   # allows the frontend to call this API from any origin

# ============================================================
# 1. Load model once at startup (not on every request)
# ============================================================
MODEL_PATH = 'macrosentinel_model.keras'

print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    model = None
    print(f"⚠ Could not load model: {e}")
    print("  Run macrosentinel_v3_final.py first to train and save the model.")

# ============================================================
# 1b. Browser-impersonating session for yfinance
#    Yahoo Finance often blocks/rate-limits requests coming from
#    cloud server IPs (Render, Railway, Streamlit Cloud, etc).
#    curl_cffi impersonates a real Chrome browser's TLS fingerprint
#    to reduce the chance of being blocked. Falls back to yfinance's
#    default session if curl_cffi isn't available.
# ============================================================
try:
    from curl_cffi import requests as cffi_requests
    YF_SESSION = cffi_requests.Session(impersonate="chrome")
    print("✅ Using curl_cffi browser-impersonating session for yfinance")
except ImportError:
    YF_SESSION = None
    print("⚠ curl_cffi not available, using yfinance's default session")

def yf_download_with_retry(sym, start, end, interval, max_retries=3):
    """Retries a yfinance download a few times with a short pause,
    since Yahoo Finance occasionally rate-limits or blips on the
    first attempt from cloud server IPs."""
    for attempt in range(1, max_retries + 1):
        try:
            kwargs = dict(start=start, end=end, interval=interval, progress=False)
            if YF_SESSION is not None:
                kwargs['session'] = YF_SESSION
            df = yf.download(sym, **kwargs)
            if df is not None and len(df) > 0:
                return df
            print(f"  ⚠ Attempt {attempt}/{max_retries} for {sym}: 0 rows, retrying...")
        except Exception as e:
            print(f"  ⚠ Attempt {attempt}/{max_retries} for {sym} failed: {e}")
        time.sleep(2)
    return pd.DataFrame()  # empty if all retries failed

# ============================================================
# 2. Helper — fetch and prepare data
# ============================================================
SYMBOLS   = ['EURUSD=X', 'GBPUSD=X']
FEATURES  = ['EURUSD=X_pct', 'GBPUSD=X_pct', 'divergence', 'news_sentiment']
TIMESTEPS = 24

def fetch_and_prepare():
    """
    Downloads the latest hourly forex data, engineers features,
    scales them, and returns everything needed for prediction.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=720)

    data = {}
    for sym in SYMBOLS:
        df = yf_download_with_retry(sym,
                                     start_date.strftime('%Y-%m-%d'),
                                     end_date.strftime('%Y-%m-%d'),
                                     '1h')
        if len(df) == 0:
            raise RuntimeError(
                f"Could not download data for {sym} after multiple retries. "
                f"Yahoo Finance may be temporarily blocking this server's IP — "
                f"try again in a moment."
            )
        df = df[['Close']].rename(columns={'Close': sym})
        data[sym] = df

    df_all = pd.concat([data[sym] for sym in SYMBOLS], axis=1).ffill()

    for sym in SYMBOLS:
        df_all[sym + '_pct'] = df_all[sym].pct_change()

    df_all['divergence'] = df_all['EURUSD=X_pct'] - df_all['GBPUSD=X_pct']
    df_all = df_all.dropna()

    # ── Dummy sentiment — replace with real NLP feed later ──
    np.random.seed(42)
    df_all['news_sentiment'] = np.random.uniform(-1, 1, len(df_all))
    # ─────────────────────────────────────────────────────────

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_all[FEATURES])

    # Build sequences
    X, y = [], []
    for i in range(TIMESTEPS, len(scaled)):
        X.append(scaled[i - TIMESTEPS:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    # 80/20 split
    split    = int(0.8 * len(X))
    X_test   = X[split:]
    y_test   = y[split:]

    # Latest raw prices for the price ticker
    latest_eur = float(df_all['EURUSD=X'].iloc[-1])
    prev_eur   = float(df_all['EURUSD=X'].iloc[-2])
    latest_gbp = float(df_all['GBPUSD=X'].iloc[-1])
    prev_gbp   = float(df_all['GBPUSD=X'].iloc[-2])

    return X_test, y_test, scaler, df_all, split, {
        'eur': latest_eur, 'eur_prev': prev_eur,
        'gbp': latest_gbp, 'gbp_prev': prev_gbp
    }

def invert_scaling(y_scaled, scaler, n_features):
    dummy       = np.zeros((len(y_scaled), n_features))
    dummy[:, 0] = y_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

def mape(actual, predicted):
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

# ============================================================
# 3. Routes
# ============================================================

@app.route('/')
def index():
    """Serve the dashboard"""
    return render_template('index.html')


@app.route('/api/prices')
def prices():
    """
    Returns the latest EUR/USD and GBP/USD prices quickly,
    without running the full model — for the live price ticker.
    """
    try:
        end   = datetime.today()
        start = end - timedelta(days=5)
        prices_data = {}
        for sym in SYMBOLS:
            df = yf_download_with_retry(sym, start.strftime('%Y-%m-%d'),
                                         end.strftime('%Y-%m-%d'), '1h')
            if len(df) < 2:
                raise RuntimeError(f"Could not download recent data for {sym}")
            prices_data[sym] = {
                'price': round(float(df['Close'].iloc[-1]), 5),
                'prev':  round(float(df['Close'].iloc[-2]), 5)
            }

        eur      = prices_data['EURUSD=X']['price']
        eur_prev = prices_data['EURUSD=X']['prev']
        gbp      = prices_data['GBPUSD=X']['price']
        gbp_prev = prices_data['GBPUSD=X']['prev']

        return jsonify({
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'EURUSD': {
                'price': eur,
                'change': round(eur - eur_prev, 5),
                'change_pct': round((eur - eur_prev) / eur_prev * 100, 4)
            },
            'GBPUSD': {
                'price': gbp,
                'change': round(gbp - gbp_prev, 5),
                'change_pct': round((gbp - gbp_prev) / gbp_prev * 100, 4)
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/predict')
def predict():
    """
    Runs the full LSTM pipeline and returns:
    - actual vs predicted pct change arrays (for the chart)
    - all evaluation metrics (for the metrics table)
    - trading signals (for the signal cards)
    - divergence values (for the divergence chart)
    """
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Run macrosentinel_v3_final.py first.'
        }), 503

    try:
        X_test, y_test, scaler, df_all, split, px = fetch_and_prepare()
        n_feat = len(FEATURES)

        # Run predictions
        y_pred = model.predict(X_test, verbose=0)

        # Inverse transform
        actual = invert_scaling(y_test,  scaler, n_feat)
        pred   = invert_scaling(y_pred,  scaler, n_feat)

        # ── Error metrics ──────────────────────────────────
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae  = float(mean_absolute_error(actual, pred))
        mape_val = mape(actual, pred)

        # ── Directional metrics ────────────────────────────
        dir_actual = (actual > 0).astype(int)
        dir_pred   = (pred   > 0).astype(int)

        da        = float(np.mean(dir_actual == dir_pred) * 100)
        precision = float(precision_score(dir_actual, dir_pred, zero_division=0) * 100)
        recall    = float(recall_score(dir_actual,    dir_pred, zero_division=0) * 100)
        f1        = float(f1_score(dir_actual,        dir_pred, zero_division=0) * 100)

        # ── Trading metrics ────────────────────────────────
        positions = np.where(dir_pred == 1, 1, 0)
        pnl       = actual * positions
        win_rate  = float(np.mean(pnl[pnl != 0] > 0) * 100) if np.any(pnl != 0) else 0.0
        mean_r, std_r = float(np.mean(pnl)), float(np.std(pnl))
        sharpe = round(mean_r / std_r * np.sqrt(6500), 4) if std_r != 0 else 0.0

        # ── Signals from latest divergence ─────────────────
        div_series = df_all['divergence'].values[TIMESTEPS + split:]
        latest_div = float(div_series[-1])
        gbp_div    = float(df_all['GBPUSD=X_pct'].values[-1] * -1)

        def signal_from_div(d):
            if d >  0.001: return 'BUY'
            if d < -0.001: return 'SELL'
            return 'HOLD'

        # ── Return last 72 points for chart ────────────────
        n_chart = min(72, len(actual))
        return jsonify({
            'status':    'ok',
            'timestamp': datetime.now().isoformat(),
            'chart': {
                'actual':    actual[-n_chart:].tolist(),
                'predicted': pred[-n_chart:].tolist(),
                'labels':    [f'T+{i}h' if i % 12 == 0 else ''
                              for i in range(n_chart)]
            },
            'divergence': {
                'values': div_series[-12:].tolist(),
                'labels': [f'{8+i}h' for i in range(min(12, len(div_series)))]
            },
            'metrics': {
                'rmse':      round(rmse, 6),
                'mae':       round(mae,  6),
                'mape':      round(mape_val, 4),
                'da':        round(da,        2),
                'precision': round(precision, 2),
                'recall':    round(recall,    2),
                'f1':        round(f1,        2),
                'win_rate':  round(win_rate,  2),
                'sharpe':    round(sharpe,    4)
            },
            'signals': {
                'EURUSD': {
                    'signal':    signal_from_div(latest_div),
                    'divergence': round(latest_div, 4)
                },
                'GBPUSD': {
                    'signal':    signal_from_div(gbp_div),
                    'divergence': round(gbp_div, 4)
                },
                'EURGBP': {
                    'signal':    signal_from_div(latest_div - gbp_div),
                    'divergence': round(latest_div - gbp_div, 4)
                }
            },
            'prices': px
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ============================================================
# 4. Run
# ============================================================
if __name__ == '__main__':
    print("\n🚀 MacroSentinel API running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)

