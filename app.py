from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# ⬇️ Load your trained model
MODEL_PATH = r"C:\Users\rp077\OneDrive\Documents\Project exhibition\code\Stock Predictions Model.keras"
model = load_model(MODEL_PATH)

# Popular tickers
POPULAR_TICKERS = [
    {'symbol': 'AAPL', 'name': 'Apple Inc.'},
    {'symbol': 'GOOG', 'name': 'Alphabet Inc.'},
    {'symbol': 'MSFT', 'name': 'Microsoft Corporation'},
    {'symbol': 'AMZN', 'name': 'Amazon.com, Inc.'},
    {'symbol': 'TSLA', 'name': 'Tesla, Inc.'},
    {'symbol': 'META', 'name': 'Meta Platforms, Inc.'},
    {'symbol': 'NFLX', 'name': 'Netflix, Inc.'},
    {'symbol': 'RELIANCE', 'name': 'Reliance Industries Limited'},
    {'symbol': 'TCS', 'name': 'Tata Consultancy Services'},
    {'symbol': 'HDFCBANK', 'name': 'HDFC Bank'},
    {'symbol': 'ICICIBANK', 'name': 'ICICI Bank'},
    {'symbol': 'BHARTIARTL', 'name': 'Bharti Airtel'},
    {'symbol': 'SBI', 'name': 'State Bank of India'},
    {'symbol': 'INFY', 'name': 'Infosys Limited'},
    {'symbol': 'AXISBANK', 'name': 'Axis Bank'},
    {'symbol': 'ITC', 'name': 'ITC Limited'},
    {'symbol': 'LT', 'name': 'Larsen & Toubro Limited'},
    {'symbol': 'COALINDIA', 'name': 'Coal India Limited'},
    {'symbol': 'IOC', 'name': 'Indian Oil Corporation'},
    {'symbol': 'ONGC', 'name': 'Oil & Natural Gas Corporation'},
    {'symbol': 'TATAMOTORS', 'name': 'Tata Motors'},
]

def plotly_to_html(fig):
    """
    Convert Plotly figure to HTML with interactive layout & Times New Roman font.
    Wider & taller for readability. Adds rangeslider and rangeselector.
    """
    fig.update_layout(
        height=650,
        width=1200,
        template="plotly_white",
        hovermode="x unified",
        font=dict(family="Times New Roman", size=15, color="#2c3e50"),
        title_font=dict(size=22, family="Times New Roman", color="#1a5276"),
        legend=dict(title="Legend", orientation="h", x=0.5, xanchor="center", y=-0.2),
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')


def get_model_lookback_and_features(keras_model, default_lookback=100):
    """
    Safely derive lookback (timesteps) and feature count from the model.
    Assumes input shape like (None, timesteps, features).
    """
    try:
        ishape = keras_model.input_shape
        # input_shape can be a list (for multi-input models); handle the common single-input case
        if isinstance(ishape, list):
            ishape = ishape[0]
        _, timesteps, features = ishape
        if timesteps is None:
            timesteps = default_lookback
        return int(timesteps), int(features)
    except Exception:
        return default_lookback, 1


def build_sequences(series_scaled: np.ndarray, lookback: int):
    """
    Turn a (N,1) scaled series into overlapping sequences for LSTM inference.
    Returns x (num_samples, lookback, 1) and y (num_samples, 1).
    """
    x, y = [], []
    for i in range(lookback, len(series_scaled)):
        x.append(series_scaled[i - lookback:i, :])
        y.append(series_scaled[i, :])
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return x, y


@app.route('/', methods=['GET', 'POST'])
def home():
    # Defaults (works even if your template doesn't yet send dates)
    stock = 'GOOG'
    start_date = '2012-01-01'
    end_date = '2022-12-31'
    data_html, graph_html, error_msg = '', {}, None

    if request.method == 'POST':
        stock_input = request.form.get('stock', '').strip().upper()
        ticker_select = request.form.get('ticker_select')
        # Optional form fields (safe defaults if not present in template)
        start_date = request.form.get('start_date', start_date)
        end_date = request.form.get('end_date', end_date)

        stock = ticker_select if ticker_select else (stock_input if stock_input else stock)

        try:
            # 1) Download data
            data = yf.download(stock, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError("No data found for this period/ticker.")

            # Flatten MultiIndex (if present)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Basic cleaning: keep standard OHLCV; ensure Close exists
            if 'Close' not in data.columns:
                raise ValueError("Downloaded data has no 'Close' column.")

            # 2) Split train/test (avoid leakage)
            split_idx = int(len(data) * 0.8)
            close_train = data[['Close']].iloc[:split_idx].copy()
            close_test = data[['Close']].iloc[split_idx:].copy()

            # Guard: need enough history to build at least one sequence
            lookback, n_features = get_model_lookback_and_features(model, default_lookback=100)
            if n_features != 1:
                raise ValueError(
                    f"Model expects {n_features} features per timestep; this app supplies 1 (Close). "
                    "Please retrain or adapt the preprocessing to match the model."
                )

            if len(close_train) < lookback + 1 or len(close_test) == 0:
                raise ValueError(
                    f"Not enough data to build sequences (need > {lookback} training points and some test data)."
                )

            # 3) Fit scaler ONLY on training data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(close_train.values)  # fit on train only

            # 4) Prepare the test block with past lookback days from train
            past_lookback = close_train.tail(lookback)
            test_block = pd.concat([past_lookback, close_test], axis=0)

            # Scale using the already-fitted scaler (no leakage)
            test_block_scaled = scaler.transform(test_block.values)  # shape (lookback + len(test), 1)

            # 5) Build sequences and targets for the test period
            x_test, y_test = build_sequences(test_block_scaled, lookback=lookback)  # shapes: (len(test), lookback, 1), (len(test),1)
            # Safety: ensure float32 for Keras
            x_test = x_test.astype(np.float32)

            # 6) Predict
            preds_scaled = model.predict(x_test, verbose=0)  # expected shape (len(test), 1)
            # 7) Inverse transform properly
            preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)
            y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

            # Dates aligned to test set
            pred_dates = close_test.index  # len(preds) should equal len(close_test)

            # 8) Build charts (MAs computed on full data for richer context)
            ma_50 = data['Close'].rolling(50).mean()
            ma_100 = data['Close'].rolling(100).mean()
            ma_200 = data['Close'].rolling(200).mean()

            # Chart 1: Price vs 50-day MA
            fig1 = go.Figure([
                go.Scatter(x=data.index, y=data['Close'], mode="lines", name="Close Price"),
                go.Scatter(x=data.index, y=ma_50, mode="lines", name="50-day MA")
            ])
            fig1.update_layout(title=f"{stock} Price vs 50-day MA")
            graph_html = {}
            graph_html['price_ma50'] = plotly_to_html(fig1)

            # Chart 2: Price vs 50 & 100-day MAs
            fig2 = go.Figure([
                go.Scatter(x=data.index, y=data['Close'], mode="lines", name="Close Price"),
                go.Scatter(x=data.index, y=ma_50, mode="lines", name="50-day MA"),
                go.Scatter(x=data.index, y=ma_100, mode="lines", name="100-day MA")
            ])
            fig2.update_layout(title=f"{stock} Price vs 50 & 100-day MAs")
            graph_html['price_ma50_ma100'] = plotly_to_html(fig2)

            # Chart 3: Price vs 100 & 200-day MAs
            fig3 = go.Figure([
                go.Scatter(x=data.index, y=data['Close'], mode="lines", name="Close Price"),
                go.Scatter(x=data.index, y=ma_100, mode="lines", name="100-day MA"),
                go.Scatter(x=data.index, y=ma_200, mode="lines", name="200-day MA")
            ])
            fig3.update_layout(title=f"{stock} Price vs 100 & 200-day MAs")
            graph_html['price_ma100_ma200'] = plotly_to_html(fig3)

            # Chart 4: Predicted vs Original (aligned on dates)
            fig4 = go.Figure([
                go.Scatter(x=pred_dates, y=preds, mode="lines", name="Predicted Price"),
                go.Scatter(x=pred_dates, y=y_true, mode="lines", name="Original Price")
            ])
            fig4.update_layout(title="Predicted vs Original Stock Prices")
            graph_html['predicted_vs_original'] = plotly_to_html(fig4)

            # Data table
            data_html = data.to_html(classes="table")

        except Exception as e:
            error_msg = f"❌ {str(e)}"

    return render_template(
        "index.html",
        stock=stock,
        popular_tickers=POPULAR_TICKERS,
        data_html=data_html,
        graph_urls=graph_html,
        error_msg=error_msg,
        start_date=start_date,
        end_date=end_date
    )


if __name__ == "__main__":
    # Tip: In production, set debug=False
    app.run(debug=True, use_reloader=False)
