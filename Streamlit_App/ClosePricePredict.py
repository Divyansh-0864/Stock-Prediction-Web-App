import streamlit as st
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import date
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
import plotly.express as px

st.title("Stock Price Prediction Using LSTM Model")
st.markdown("---")
#Load The Required Stock Data
startDate = pd.to_datetime('2012-01-01')
endDate = pd.to_datetime(date.today())

company = st.selectbox("Select Stock ticker",options=("AAPL","MSFT","TSLA","AMZN","NVDA"))
df = web.DataReader(company, data_source = 'stooq', start = startDate, end = endDate)
df = df.reindex(index = df.index[::-1])

# Describing the data
st.markdown("### Data from 2012 - Today")
st.table(df.describe())

# Last 5 days
st.markdown("## Last 5 days data")
st.table(df.tail())

#Visualize the Closing Stock Price History
st.markdown("### Closing Price vs Time Chart")
fig = plt.figure(figsize=(12,6))
plt.title("Closing Stock Price")
plt.xlabel("Date")
plt.ylabel("CLosing Stock Price ($)")
plt.plot(df['Close'])
st.pyplot(fig)
st.markdown("---")

# Rolling average of 100 days
st.markdown("### Moving Average(100 & 200 days) vs Time Chart")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)
st.markdown("---")

#Create A New DataFrame with only Close column
data = df.filter(['Close'])
#Convert DataFrame To Numpy Array
dataset = data.values
# Training -> 80% data
# Testing -> 20% data
training_data_len = math.ceil((len(dataset)) * 0.8)
print("Training data length =", end = " ")
print(training_data_len)

# Scale the Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# Create scaled training dataset
train_data = scaled_data[0:training_data_len]
x_train,y_train = [],[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

# Convert the x_train and y_train to numpy array
x_train,y_train = np.array(x_train),np.array(y_train)
# Resize the arrays
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# Unscaled data
unscaled_data = scaler.inverse_transform(dataset)

# Build the Model
model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1))) # input layer
model.add(LSTM(50,return_sequences=False)) # Long term memory lane
model.add(Dense(25)) # Short term memory lane
model.add(Dense(1)) # Forget gate

# Compilation of Model
model.compile(optimizer='adam',loss="mse")
# Fitting The model on trainig set
model.fit(x_train,y_train,batch_size=1,epochs=2)

# Create a Testing Dataset
test_data = scaled_data[training_data_len-60:,:]
x_test = []
y_test = dataset[training_data_len :, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert Data to numpy array
x_test = np.array(x_test)
# Reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# Get the model predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean((predictions-y_test)**2))
print("Error % = ", rmse)
print("Accuracy % = ", (100-rmse))

# Plot the Data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

st.markdown("## Predictions")
predicted_fig = plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Date")
plt.ylabel("Closing Stock Price")
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
st.pyplot(predicted_fig)

st.write("Error =",rmse)
st.write("Accuracy =",(100-rmse))

# Using the model to predict future price
Co_quote = web.DataReader(company,data_source='stooq',start=startDate,end=endDate)
new_df = Co_quote.filter(['Close'])
last_60_days = new_df[:60].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
st.write("### Predicted price for next day =",pred_price[0,0])

st.markdown("---")
st.markdown("### Intreactive Graph")
# Plot the Data using Plotly
fig = px.line(valid, x=valid.index, y=['Close', 'Predictions'],labels={'index': 'Date', 'value': 'Price'},title='Stock Price Prediction')
st.plotly_chart(fig)