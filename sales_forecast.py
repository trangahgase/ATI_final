

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

store_sales = pd.read_csv('train.csv')
store_sales.head()


store_sales.info()


store_sales = store_sales.drop(['store', 'item'], axis = 1)
store_sales.info()



store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales.info()



store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()

monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales.head()

plt.figure(figsize = (15, 5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title("Monthtly Customer Sales")
plt.show()

monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
#let's also use the dropna to drop all the NaN values that can be returned form the diff function above.
monthly_sales = monthly_sales.dropna()

supervised_data = monthly_sales.drop(['date','sales'], axis=1)



for i in range (1, 13):
    col_name = 'month_'+ str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)

supervised_data = supervised_data.dropna().reset_index(drop=True)
supervised_data.head(10)



train_data = supervised_data[:-12] #previous 12
test_data = supervised_data[-12:] #next 12

print("Train data shape: ", train_data.shape)
print("Test data shape: ", test_data.shape)

scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)
joblib.dump(scaler, 'scaler.pkl')


X_train, y_train = train_data[:,1:], train_data[:,0:1]
X_test, y_test = test_data[:,1:], test_data[:0,:1]
y_train = y_train.ravel()
y_test = y_test.ravel()


print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)



sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

predict_df.head()

act_sales = monthly_sales['sales'][-13:].to_list()

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
joblib.dump(lr_model, 'lr_model.pkl')



lr_pred = lr_pred.reshape(-1,1)
lr_pred_test_set = np.concatenate([lr_pred, X_test], axis=1)
lr_pred_test_set = scaler.inverse_transform(lr_pred_test_set)

result_list = []
for index in range (0, len(lr_pred_test_set)):
    result_list.append(lr_pred_test_set [index][0] + act_sales[index])

lr_pred_series = pd.Series(result_list, name = "Linear Prediction")
predict_df = predict_df.merge(lr_pred_series, left_index=True, right_index=True)


lr_mse = np.sqrt(mean_squared_error(predict_df["Linear Prediction"], monthly_sales['sales'][-12:]))
lr_mae = mean_absolute_error(predict_df["Linear Prediction"], monthly_sales['sales'][-12:])
lr_r2 = r2_score(predict_df["Linear Prediction"], monthly_sales['sales'][-12:])
print('Linear Regression MSE: ', lr_mse)
print('Linear Regression MAE: ', lr_mae)
print('Linear Regression R2: ',lr_r2)

plt.figure(figsize=(15,5))

plt.plot(monthly_sales['date'], monthly_sales['sales'])

plt.plot(predict_df['date'], predict_df['Linear Prediction'])
plt.title('Customer sales Forecast using LR Model')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(['Actual sales', 'Predicted Sales'])
plt.show()



plt.figure(figsize=(15,5))
plt.bar(monthly_sales['date'], monthly_sales['sales'], width=10)
plt.plot(predict_df['date'], predict_df['Linear Prediction'], color = 'red')
plt.xlabel("Date")
plt.ylabel('Sales')

plt.title('Predictions vs Actual')
plt.show()




