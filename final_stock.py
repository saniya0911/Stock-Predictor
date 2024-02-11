import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
# pd.options.plotting.backend = 'plotly'
import plotly.graph_objects as go

# Set the page title

st.title("Stock Price Prediction App")

# Define the list of companies

tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NFLX', 'UBER']

# Display the company selection dropdown

selected_option = st.selectbox("Select a company", tech_list)

# Get the number of days to predict from the user

future_days = st.number_input("Enter the number of days to predict", min_value=1, value=7,max_value=10)

# Get the historical stock data

end_date = datetime.now()
start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
data = yf.download(selected_option, start=start_date, end=end_date)
closed_df = pd.DataFrame(data['Close'])

# Create feature and target datasets

df = closed_df.reset_index()
df['Prediction'] = df[['Close']].shift(-future_days)
X = np.array(df.drop(['Prediction', 'Date'], axis=1))[:-future_days]
y = np.array(df['Prediction'])[:-future_days]
past_days = df['Close'].tail(100)

# Split the data into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



# Train the prediction model

rbf_svr = SVR(kernel='rbf', C=10.0)
rbf_svr.fit(x_train, y_train)

# Calculate accuracy

accuracy = rbf_svr.score(x_test, y_test)
st.write("Accuracy:", accuracy)


# Make predictions for future days

x_future = df.drop(['Prediction', 'Date'], axis=1).tail(future_days)
x_future = np.array(x_future)
predicted_values = rbf_svr.predict(x_future)

present_day= df.drop(['Prediction','Date'],axis=1)
present_day=np.array(present_day.tail(1))

# Visualize the data

values = np.insert(predicted_values, 0, present_day, axis=0)
predictions = pd.DataFrame(values)
predictions.index += len(past_days) - 1

#plt.figure(figsize=(12,8))
##plt.title(selected_option + " Stock Price Prediction")
#plt.xlabel('Days')
#plt.ylabel('Closed Price')
#plt.plot(df['Close'])
#plt.plot(predictions)
#plt.legend(['Original', 'Predicted'])
#st.pyplot(plt)
# st.plotly_chart(plt, use_container_width=True)

#Visualize data using PLotly
fig = go.Figure()

#original closing prices
fig.add_trace(go.Scatter(x=df.index, y=past_days, mode='lines', name='Original'))

#predicted closing prices
fig.add_trace(go.Scatter(x=predictions.index, y=predictions[0], mode='lines', name='Predicted'))

fig.update_layout(
    title=selected_option + " Stock Price Prediction",
    xaxis=dict(title='Days'),
    yaxis=dict(title='Closed Price'),
    legend=dict(x=0, y=1, traceorder='normal')
)
st.plotly_chart(fig)

# Make predictions for future days
#x_future_dates = df['Date'].tail(future_days)  # Extracting dates for future predictions
#x_future = df.drop(['Prediction'], axis=1).tail(future_days)
#predicted_values = rbf_svr.predict(np.array(x_future.drop(['Date'], axis=1)))

# Create a DataFrame with predicted values and corresponding dates
predicted_df = pd.DataFrame({
    #'Date': x_future_dates.values,  # Using values to get a simple array from the pandas Series
    'Predicted Close': predicted_values
})

# Display predicted values with dates on Streamlit
#st.write("Predicted Stock Prices for the Next", future_days, "Days")
#st.write(predicted_df)