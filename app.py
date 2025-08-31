from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained Keras model (update path if needed)
model = load_model(r'C:\Users\rp077\OneDrive\Documents\Project exhibition\code\Stock Predictions Model.keras')

def plot_to_img(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def home():
    graph_urls = {}
    stock = 'GOOG'
    data_html = ''
    prediction_plot_url = None

    if request.method == 'POST':
        stock = request.form['stock'].upper()

        # Download data
        start = '2012-01-01'
        end = '2022-12-31'
        data = yf.download(stock, start, end)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data_html = data.to_html(classes='table table-striped')

        data_train = pd.DataFrame(data.Close[0:int(len(data)*0.8)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.8): len(data)])

        scaler = MinMaxScaler(feature_range=(0,1))
        past_100_days = data_train.tail(100)
        data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
        data_test_scaled = scaler.fit_transform(data_test_full)

        # Moving averages
        ma_50 = data.Close.rolling(50).mean()
        ma_100 = data.Close.rolling(100).mean()
        ma_200 = data.Close.rolling(200).mean()

        # Plot Price vs MA50
        fig1 = plt.figure(figsize=(8,6))
        plt.plot(ma_50, 'r', label='MA50')
        plt.plot(data.Close, 'g', label='Close Price')
        plt.title('Stock Price vs 50-day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        graph_urls['price_ma50'] = plot_to_img(fig1)

        # Plot Price vs MA50 vs MA100
        fig2 = plt.figure(figsize=(8,6))
        plt.plot(ma_50, 'r', label='MA50')
        plt.plot(ma_100, 'b', label='MA100')
        plt.plot(data.Close, 'g', label='Close Price')
        plt.title('Stock Price vs 50-day and 100-day Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        graph_urls['price_ma50_ma100'] = plot_to_img(fig2)

        # Plot Price vs MA100 vs MA200
        fig3 = plt.figure(figsize=(8,6))
        plt.plot(ma_100, 'r', label='MA100')
        plt.plot(ma_200, 'b', label='MA200')
        plt.plot(data.Close, 'g', label='Close Price')
        plt.title('Stock Price vs 100-day and 200-day Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        graph_urls['price_ma100_ma200'] = plot_to_img(fig3)

        # Prepare test data for prediction
        x_test = []
        y_test = []
        for i in range(100, data_test_scaled.shape[0]):
            x_test.append(data_test_scaled[i-100:i])
            y_test.append(data_test_scaled[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        # Make predictions
        predictions = model.predict(x_test)
        scale_factor = 1/scaler.scale_[0]

        predictions_rescaled = predictions * scale_factor
        y_test_rescaled = y_test * scale_factor

        # Plot predicted vs original price
        fig4 = plt.figure(figsize=(8,6))
        plt.plot(predictions_rescaled, 'r', label='Predicted Price')
        plt.plot(y_test_rescaled, 'g', label='Original Price')
        plt.title('Predicted vs Original Stock Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        graph_urls['predicted_vs_original'] = plot_to_img(fig4)

    return render_template('index.html', stock=stock, data_html=data_html, graph_urls=graph_urls)

if __name__ == '__main__':
    app.run(debug=True)
