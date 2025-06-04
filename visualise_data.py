import numpy as np
import pandas as pd
import plotly.graph_objects as go

from process_data import present_day, past_dataset

def visualise(predicted_values, df, selected_option):
    today = present_day(df = df)    
    past_days = past_dataset(df= df, days= 100)

    values = np.insert(predicted_values, 0, today, axis=0)
    predictions = pd.DataFrame(values)
    predictions.index += len(past_days) - 1

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
    return fig