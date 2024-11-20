from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'
DATA_FILE = "train.csv"

# Load pre-trained model and scaler
lr_model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")


def generate_plot(monthly_sales, predictions=None):
    plt.figure(figsize=(15, 5))
    plt.plot(monthly_sales['date'], monthly_sales['sales'], label='Actual Sales', color='blue')
    if predictions is not None:
        plt.plot(monthly_sales['date'][-len(predictions):], predictions, label='Predicted Sales', color='red',
                 linestyle='--')
    plt.title("Customer Sales Forecast")
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid()
    plt.savefig('static/sales_forecast.png')
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    mse, mae, r2 = None, None, None  # Initialize metrics to None

    if request.method == 'POST':
        # Handling CSV upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)

            file.save(DATA_FILE)
            store_sales = pd.read_csv(DATA_FILE)
            store_sales['date'] = pd.to_datetime(store_sales['date'])
            store_sales = store_sales.drop(['store', 'item'], axis=1)

            monthly_sales = store_sales.groupby(store_sales['date'].dt.to_period('M'))['sales'].sum().reset_index()
            monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

            monthly_sales['sales_diff'] = monthly_sales['sales'].diff().dropna()
            monthly_sales = monthly_sales.dropna()

            supervised_data = monthly_sales[['sales_diff']].copy()
            for i in range(1, 13):
                supervised_data[f'month_{i}'] = supervised_data['sales_diff'].shift(i)
            supervised_data = supervised_data.dropna().reset_index(drop=True)

            train_data = supervised_data[:-12]
            test_data = supervised_data[-12:]

            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

            x_train, y_train = train_data[:, 1:], train_data[:, 0:1].ravel()
            x_test, y_test = test_data[:, 1:], test_data[:, 0:1].ravel()

            lr_model = LinearRegression()
            lr_model.fit(x_train, y_train)
            lr_predictions = lr_model.predict(x_test)
            lr_predictions = lr_predictions.reshape(-1, 1)
            lr_pre_test_set = np.concatenate([lr_predictions, x_test], axis=1)
            lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

            actual_sales = monthly_sales['sales'][-13:].to_list()
            result_list = [lr_pre_test_set[i][0] + actual_sales[i] for i in range(len(lr_pre_test_set))]
            lr_pre_series = pd.Series(result_list, name='Linear_Prediction')

            mse = np.sqrt(mean_squared_error(result_list, monthly_sales['sales'][-12:]))
            mae = mean_absolute_error(result_list, monthly_sales['sales'][-12:])
            r2 = r2_score(result_list, monthly_sales['sales'][-12:])

            generate_plot(monthly_sales, predictions=lr_pre_series)

    return render_template('index.html', mse=mse, mae=mae, r2=r2)


@app.route('/plot')
def plot():
    return send_file('static/sales_forecast.png', mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)