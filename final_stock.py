import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go


from process_data import download_previous_data, feature_dataset, target_dataset, future_dataset
from svm_model import split_data, train_model, calculate_accuracy, predict
from visualise_data import visualise

# Set the page title
st.title("Stock Price Prediction App")

# Define the list of companies
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NFLX', 'UBER']

# Display the company selection dropdown
selected_option = st.selectbox("Select a company", tech_list)

# Get the number of days to predict from the user
future_days = st.number_input("Enter the number of days to predict", min_value=1, value=7,max_value=10)

# Get the historical stock data
df = download_previous_data(period="1y", selected_option= selected_option, future_days=future_days)

# Create feature and target datasets

X = feature_dataset(df = df, future_days= future_days)
y = target_dataset(df= df, future_days= future_days)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = split_data(X, y, test_size=0.25)

# Train the prediction model
rbf_svr = train_model('rbf',10.0,x_train, y_train)

# Calculate accuracy

accuracy = calculate_accuracy(svr=rbf_svr, x_test= x_test, y_test = y_test)
st.write("Accuracy:", accuracy)


# Make predictions for future days

x_future = future_dataset(df= df, future_days= future_days)
predicted_values = predict(model = rbf_svr, x_future= x_future)

# Visualize the data

fig = visualise(predicted_values=predicted_values, df = df, selected_option=selected_option)
st.plotly_chart(fig)
