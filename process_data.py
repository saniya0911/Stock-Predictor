import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd


def download_previous_data(period, selected_option, future_days):
    ticker = yf.Ticker(selected_option)
    # data = yf.download(ticker, start=start_date, end=end_date)
    data = ticker.history(period=period)
    closed_df = pd.DataFrame(data['Close'])
    df = closed_df.reset_index()
    df['Prediction'] = df[['Close']].shift(-future_days)
    return df

def feature_dataset(df, future_days):
    X = np.array(df.drop(['Prediction', 'Date'], axis=1))[:-future_days]
    return X

def target_dataset(df, future_days):
    y = np.array(df['Prediction'])[:-future_days]
    return y

def past_dataset(df, days):
    past_days = df['Close'].tail(days)
    return past_days

def future_dataset(df, future_days):
    x_future = df.drop(['Prediction', 'Date'], axis=1).tail(future_days)
    x_future = np.array(x_future)
    return x_future

def present_day(df):
    present_day= df.drop(['Prediction','Date'],axis=1)
    present_day=np.array(present_day.tail(1))
    return present_day