######################## Imports #############################

# Oanda Imports
from turtle import width
import v20
from v20.order import MarketOrderRequest

# Data Imports
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import namedtuple, deque
import os
import time
import json
import pytz

# Numba Imports
from numba import njit
from numba import prange

# Trading Imports
import talib
import vectorbtpro as vbt

# Dash Imports
import keyboard
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import threading
import atexit
import dash_bootstrap_components as dbc

stop_event = threading.Event()
lock = threading.Lock()

##################### Pushover iPhone ########################

import requests

def send_pushover_notification(user_key, api_token, message, title=None):
    url = "https://api.pushover.net/1/messages.json"
    
    data = {
        'token': api_token,
        'user': user_key,
        'message': message
    }
    
    if title:
        data['title'] = title

    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        print("Notification sent successfully!")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}, Response: {response.text}")

# Replace with your actual user key and API token
YOUR_USER_KEY = 'uspdtom1ciewmn45ztxvqxkc4iidvk'
YOUR_API_TOKEN = 'atopfyfrcfnjj35j77hshwuyb4dq6m'

##################### Timezone ###############################

# Create a timezone object for EST
eastern = pytz.timezone('US/Eastern')

def adjust_fractional_seconds(timestamp):
    # Extract the fractional seconds part of the timestamp
    fractional_seconds = timestamp.split('.')[1].split('Z')[0]
    
    # Adjust the fractional seconds to always have a 6-digit precision
    while len(fractional_seconds) < 6:
        fractional_seconds += '0'
    fractional_seconds = fractional_seconds[:6]
    
    # Reconstruct the timestamp with adjusted fractional seconds
    adjusted_timestamp = timestamp.split('.')[0] + '.' + fractional_seconds + 'Z'
    return adjusted_timestamp

##################### Redis ##################################

import redis
    
redis_connection_1 = redis.Redis()
redis_connection_2 = redis.Redis()
redis_connection_3 = redis.Redis()
redis_connection_1.flushall()

def get_redis_connection(id):
    if id == 1:
        return redis_connection_1
    elif id == 2:
        return redis_connection_2
    elif id == 3:
        return redis_connection_3
    else:
        raise ValueError('Invalid connection ID')

def decode_data(r):
    data = r.get('data')
    if data is not None:
        data = json.loads(data)
        df = pd.DataFrame(data)
        if 'Open time' in df.columns:
            df['Open time'] = pd.to_datetime(df['Open time'], format="%Y-%m-%dT%H:%M:%S.%f%z")
            df['Open time'] = df['Open time'].dt.tz_convert('US/Eastern')
        return df
    return pd.DataFrame()

##################### Dash App #################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Graph(id='live-graph'),
])

@atexit.register
def stop_streaming():
    r = get_redis_connection(2)
    r.set('stop_streaming', 'True')
    r.flushall()
    redis_connection_1.close() # Close the Redis connections
    redis_connection_2.close()
    stop_event.set() # Set the stop event    
    
@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    
    # Create a new Redis connection
    r = get_redis_connection(1)

    with lock:        
        # Get Data From Redis
        df = decode_data(r)
        
    if df is None:
        return go.Figure()
    
    required_columns_initial = [
       'Open time', 'Open', 'High', 'Low', 'Close'
    ]
    
    # Create empty Figure
    fig = go.Figure()

    if all(col in df.columns for col in required_columns_initial):
        
        # Filter data for last 1000 minutes
        now = pd.Timestamp.now(tz='US/Eastern')
        cutoff = now - pd.Timedelta(minutes=1000)
        df = df.loc[df['Open time'] > cutoff]
        
        df_candles = df.drop_duplicates(subset=['Open time'], keep='last')

        fig.add_trace(go.Candlestick(
            x=df_candles['Open time'],
            open=df_candles['Open'],
            high=df_candles['High'],
            low=df_candles['Low'],
            close=df_candles['Close'],
            increasing_line_color='black',
            decreasing_line_color='black',
            increasing_fillcolor='white',
            decreasing_fillcolor='black',
            line=dict(width=0.5),
            name="Historical"
        ))

    # Column names you are looking for
    required_columns = [
        'short entry 1', 'short entry 2', 'short entry 3', 
        'long entry 1', 'long entry 2', 'long entry 3', 
        'short stop-loss hit', 'long stop-loss hit', 
        'short take-profit hit', 'long take-profit hit', 
        'bids', 'asks', 'short cond', 'long cond',
        'short entry 1 level', 'short entry 2 level', 
        'short entry 3 level', 'long entry 1 level', 
        'long entry 2 level', 'long entry 3 level', 
        's1', 's2', 's3', 'short stop-loss', 'short take-profit', 
        'l1', 'l2', 'l3', 'long stop-loss', 'long take-profit'
    ]
    
    if all(col in df.columns for col in required_columns):
        
        # Define level_colors
        long_colors = ['limegreen', 'green', 'darkgreen']
        short_colors = ['red', 'firebrick', 'darkred']
        
        for i in range(1, 4):
            if df[f'short entry {i}'].fillna(False).any():
                fig.add_trace(go.Scatter(
                    x=df.loc[df[f'short entry {i}'].fillna(False), 'Open time'], 
                    y=df.loc[df[f'short entry {i}'].fillna(False), 'bids'], 
                    mode='markers',
                    marker=dict(color=short_colors[i-1], symbol='triangle-down', size=10), # down-triangles for short
                    name=f'Short entry {i}'
                ))
            if df[f'long entry {i}'].fillna(False).any():
                fig.add_trace(go.Scatter(
                    x=df.loc[df[f'long entry {i}'].fillna(False), 'Open time'], 
                    y=df.loc[df[f'long entry {i}'].fillna(False), 'asks'], 
                    mode='markers',
                    marker=dict(color=long_colors[i-1], symbol='triangle-up', size=10), # up-triangles for long
                    name=f'Long entry {i}'
                ))

        if df['short stop-loss hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['short stop-loss hit'].fillna(False), 'Open time'], 
                y=df.loc[df['short stop-loss hit'].fillna(False), 'asks'], 
                mode='markers',
                marker=dict(color='darkviolet', size=10), 
                name='Short stop-loss hit'
            ))

        if df['long stop-loss hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['long stop-loss hit'].fillna(False), 'Open time'], 
                y=df.loc[df['long stop-loss hit'].fillna(False), 'bids'], 
                mode='markers',
                marker=dict(color='darkviolet', size=10), 
                name='Long stop-loss hit'
            ))

        if df['short take-profit hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['short take-profit hit'].fillna(False), 'Open time'], 
                y=df.loc[df['short take-profit hit'].fillna(False), 'asks'], 
                mode='markers',
                marker=dict(color='dodgerblue', size=10), 
                name='Short take-profit hit'
            ))

        if df['long take-profit hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['long take-profit hit'].fillna(False), 'Open time'], 
                y=df.loc[df['long take-profit hit'].fillna(False), 'bids'], 
                mode='markers',
                marker=dict(color='dodgerblue', size=10), 
                name='Long take-profit hit'
            ))

        for i in range(1, 4):
            y_values = df[f'short entry {i} level'].where(df['short cond']).tolist()
            x_values = df['Open time'].tolist()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line_shape='hv',
                line=dict(color=short_colors[i-1]),
                name=f'Short entry {i} level'
            ))
            
        for i in range(1, 4):
            y_values = df[f'long entry {i} level'].where(df['long cond']).tolist()
            x_values = df['Open time'].tolist()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line_shape='hv',
                line=dict(color=long_colors[i-1]),
                name=f'Long entry {i} level'
            ))
                
        short_condition = df['s1'] | df['s2'] | df['s3']
        fig.add_trace(go.Scatter(
            x=df.loc[short_condition, 'Open time'],
            y=df.loc[short_condition, 'short stop-loss'],
            mode='lines',
            line_shape='hv',
            line=dict(color='darkviolet'),
            name='Short stop-loss'
        ))
        fig.add_trace(go.Scatter(
            x=df.loc[short_condition, 'Open time'],
            y=df.loc[short_condition, 'short take-profit'],
            mode='lines',
            line_shape='hv',
            line=dict(color='dodgerblue'),
            name='Short take-profit'
        ))

        long_condition = df['l1'] | df['l2'] | df['l3']
        fig.add_trace(go.Scatter(
            x=df.loc[long_condition, 'Open time'],
            y=df.loc[long_condition, 'long stop-loss'],
            mode='lines',
            line_shape='hv',
            line=dict(color='darkviolet'),
            name='Long stop-loss'
        ))
        fig.add_trace(go.Scatter(
            x=df.loc[long_condition, 'Open time'],
            y=df.loc[long_condition, 'long take-profit'],
            mode='lines',
            line_shape='hv',
            line=dict(color='dodgerblue'),
            name='Long take-profit'
        ))
        
        ################### Moving Averages ######################
        
        fig.add_trace(go.Scatter(
                x=df.loc['ma fast'],
                y=df.loc['ma fast'],
                mode='lines',
                name='MA fast',
                line=dict(color='limegreen', width=1)
            )
        )
        fig.add_trace(go.Scatter(
                x=df.loc['ma mid'],
                y=df.loc['ma mid'],
                mode='lines',
                name='ma mid',
                line=dict(color='yellow', width=1)
            )
        )
        fig.add_trace(go.Scatter(
                x=df.loc['ma slow'],
                y=df.loc['ma slow'],
                mode='lines',
                name='ma slow',
                line=dict(color='red', width=1)
            )
        )


    # Laptop Dimensions (Chrome) Width: 1365 Height: 923
    # Desktop Dimensions (Chrome) Height: 1291
    fig.update_layout(
        plot_bgcolor='lightgrey',
        paper_bgcolor='lightgrey',
        font=dict(color='black'),
        hovermode='x',
        autosize=True,  # Set autosize to False
        #width=1365,      # Set the fixed width
        height=1291,      # Set the fixed height
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            zeroline=False,
            showticklabels=True,
            showgrid=False,
            rangeslider=dict(visible=False) 
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    
    return fig

##################### API Authentication ########################

# Authenticate to OANDA
access_token = '650d35fdf3fdb30d1056bcfe6aa9ae24-7ba8cbb7e3759f01e7b8b2775f7d9b8d'
account_id = '001-001-8347571-001'
hostname = 'api-fxtrade.oanda.com'
streaming_hostname = 'stream-fxtrade.oanda.com'
instrument = 'EUR_USD'

# Create an API context
api = v20.Context(hostname, 443, token=access_token)
streaming_api = v20.Context(streaming_hostname, 443, token=access_token)

##################### Fetch Candles #############################

def fetch_candles(instrument, granularity, count):
    delay = 0.5     # Seconds to wait between retries

    while True:
        try:
            params = {
                "granularity": granularity,
                "count": count,
            }

            # Fetch the candles
            response = api.instrument.candles(instrument, **params)

            if response.status != 200:
                print(f"Error: {response.body}")
                return None

            candles = response.body.get("candles", [])
            if not candles:
                print("No candle data found")
                return None

            df = pd.DataFrame([{
                "Open time": pd.to_datetime(candle.time),
                "Open": float(candle.mid.o),
                "High": float(candle.mid.h),
                "Low": float(candle.mid.l),
                "Close": float(candle.mid.c),
                "Volume": int(candle.volume),
                "Complete": bool(candle.complete)
            } for candle in candles[:-1]])  # Exclude the last candle

            df['Open time'] = df['Open time'].dt.tz_convert('US/Eastern')
            df = df.set_index('Open time')  # Set the 'Open time' column as the index

            # Convert granularity to pandas frequency
            freq_map = {'M1': '1min', 'M5': '5min'}
            freq = freq_map.get(granularity, granularity)
            
            # Resample to the granularity and forward-fill any gaps
            df = df.resample(freq).ffill()

            # Fill any remaining NaN values (optional)
            df = df.fillna(method='ffill')

            # Convert columns to float64 if needed (optional)
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(np.float64)

            return df

        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)       # Wait for 5 seconds before retrying

##################### Ensure 2-Dim Arrays #######################

def ensure_2d(arr):
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

######################### Individual Test Runs #########################    
'''
Top trial information from the last study:

Objective values:
Objective 1: 0.0165531
Objective 2: 232.0
Objective 3: 392.7513514

Constraint values:
Constraint 1: -0.01028694035294329
Constraint 2: -87.24864864864867
Constraint 3: -32.0
Constraint 4: -0.016553146113888095

2023-07-10 00:55	2023-10-13 21:00
'''
first_target_length_input = 110
rsi_lower_threshold = 35
rsi_period = 7
rsi_upper_threshold = 62
sl_multiplier = 16.4
sl_period = 870
ma_fast_period = 21
ma_mid_period = 50
ma_slow_period = 200
#st_multiplier = 4.2
#st_period = 136
short_target_multiple = 0.00000
long_target_multiple  = 0.00000
short_entry_multiple  = 0.00000
long_entry_multiple   = 0.00000
will_frac_period = 2

##################### Indicator Functions #######################

##################### ATR (Stop Loss) ###########################
def get_sl_atr(high, low, close, period, multiplier):
    sl_atr = vbt.talib('ATR').run(high, low, close, timeperiod=period, skipna=True).real.to_numpy().astype(np.float64)
    short_sl_atr_value, long_sl_atr_value = get_sl_atr_nb(sl_atr, close, multiplier)
    return short_sl_atr_value, long_sl_atr_value

@njit(nogil=True)
def get_sl_atr_nb(sl_atr, close, multiplier):
    sl_scaled_atr = sl_atr * multiplier
    short_sl_atr_value = close + sl_scaled_atr
    long_sl_atr_value = close - sl_scaled_atr
    return short_sl_atr_value, long_sl_atr_value
'''
####### SuperTrend Indicator Helper Functions ###################
def st_get_basic_bands(high, low, close, st_multiplier, st_period):
    st_medprice = vbt.talib('MEDPRICE').run(high, low, skipna=True).real.to_numpy().astype(np.float64)
    st_atr = vbt.talib('ATR').run(high, low, close, timeperiod=st_period, skipna=True).real.to_numpy().astype(np.float64)
    st_basic_upper, st_basic_lower = st_get_basic_bands_nb(st_atr, st_medprice, st_multiplier, st_period)
    return st_basic_upper, st_basic_lower

@njit(nogil=True)
def st_get_basic_bands_nb(st_atr, st_medprice, st_multiplier, st_period):
    st_matr = st_multiplier * st_atr
    st_basic_upper = st_medprice + st_matr
    st_basic_lower = st_medprice - st_matr
    return st_basic_upper, st_basic_lower

@njit(nogil=True)
def st_get_final_bands(close, st_basic_upper, st_basic_lower): 
    
    st_trend = np.full(close.shape, np.nan)  
    st_dir = np.full(close.shape, 1)
    st_long = np.full(close.shape, np.nan)
    st_short = np.full(close.shape, np.nan)
    
    for i in range(1, close.shape[0]):
        if close[i] > st_basic_upper[i - 1]:
            st_dir[i] = 1
        elif close[i] < st_basic_lower[i - 1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i - 1]
            if st_dir[i] > 0 and st_basic_lower[i] < st_basic_lower[i - 1]:
                st_basic_lower[i] = st_basic_lower[i - 1]
            if st_dir[i] < 0 and st_basic_upper[i] > st_basic_upper[i - 1]:
                st_basic_upper[i] = st_basic_upper[i - 1]
        if st_dir[i] > 0:
            st_trend[i] = st_long[i] = st_basic_lower[i]
        else:
            st_trend[i] = st_short[i] = st_basic_upper[i]    
    return st_trend, st_dir, st_long, st_short 
'''
##################### RSI Indicator #############################
def get_rsi(close, period, upper_threshold, lower_threshold):
    rsi = vbt.talib('RSI').run(close, timeperiod=period, skipna=True).real.to_numpy().astype(np.float64)
    above_upper_threshold, below_lower_threshold = get_rsi_nb(rsi, upper_threshold, lower_threshold)
    return above_upper_threshold, below_lower_threshold

@njit(nogil=True)
def get_rsi_nb(rsi, upper_threshold, lower_threshold):
    above_upper_threshold = rsi >= upper_threshold
    below_lower_threshold = rsi <= lower_threshold
    return above_upper_threshold, below_lower_threshold

######### Williams Fractals Helper Functions ###################
def find_highest_lowest(high, low, first_target_length_input):
    highest_of_past_length = vbt.talib('MAX').run(high, timeperiod=first_target_length_input, skipna=True).real.to_numpy().astype(np.float64)
    lowest_of_past_length = vbt.talib('MIN').run(low, timeperiod=first_target_length_input, skipna=True).real.to_numpy().astype(np.float64)
    return highest_of_past_length, lowest_of_past_length


@njit(nogil=True)
def will_frac(high, 
              low, 
              highest_of_past_length, 
              lowest_of_past_length, 
              fractal_period, 
              long_target_multiple, 
              short_target_multiple, 
              long_entry_multiple, 
              short_entry_multiple):
    
    botFractals = np.full(low.shape, False)
    topFractals = np.full(high.shape, False)

    long_take_profit_value = np.full(high.shape, np.nan)
    short_take_profit_value = np.full(low.shape, np.nan)

    short_entry_max_value = np.full(high.shape, np.nan)
    short_entry_min_value = np.full(high.shape, np.nan)
    short_entry_mid_value = np.full(high.shape, np.nan)    

    long_entry_max_value = np.full(low.shape, np.nan)
    long_entry_min_value = np.full(low.shape, np.nan)
    long_entry_mid_value = np.full(low.shape, np.nan)

    short_entry_first = np.nan
    short_entry_second = np.nan
    short_entry_third = np.nan

    long_entry_first = np.nan
    long_entry_second = np.nan
    long_entry_third = np.nan

    for n in range(1, high.shape[0]): # range(1, len(h)):

        upflagDownFrontier = True
        upflagUpFrontier0 = True
        upflagUpFrontier1 = True
        upflagUpFrontier2 = True
        upflagUpFrontier3 = True
        upflagUpFrontier4 = True

        downflagDownFrontier = True
        downflagUpFrontier0 = True
        downflagUpFrontier1 = True
        downflagUpFrontier2 = True
        downflagUpFrontier3 = True
        downflagUpFrontier4 = True

        for i in range(1, fractal_period + 1):
            upflagDownFrontier = upflagDownFrontier and (high[n-(fractal_period-i)] < high[n-fractal_period])
            upflagUpFrontier0 = upflagUpFrontier0 and (high[n-(fractal_period+i)] < high[n-fractal_period])
            upflagUpFrontier1 = upflagUpFrontier1 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+i+1)] < high[n-fractal_period])                                                       
            upflagUpFrontier2 = upflagUpFrontier2 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+2)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+i+2)] < high[n-fractal_period])                                                   
            upflagUpFrontier3 = upflagUpFrontier3 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+2)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+3)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+i+3)] < high[n-fractal_period])
            upflagUpFrontier4 = upflagUpFrontier4 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+2)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+3)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+4)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+i+4)] < high[n-fractal_period])

            downflagDownFrontier = downflagDownFrontier and (low[n-(fractal_period-i)] > low[n-fractal_period])
            downflagUpFrontier0 = downflagUpFrontier0 and (low[n-(fractal_period+i)] > low[n-fractal_period])
            downflagUpFrontier1 = downflagUpFrontier1 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+i+1)] > low[n-fractal_period])          
            downflagUpFrontier2 = downflagUpFrontier2 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+2)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+i+2)] > low[n-fractal_period])
            downflagUpFrontier3 = downflagUpFrontier3 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+2)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+3)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+i+3)] > low[n-fractal_period])
            downflagUpFrontier4 = downflagUpFrontier4 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+2)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+3)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+4)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+i+4)] > low[n-fractal_period])

        flagUpFrontier = (upflagUpFrontier0 or upflagUpFrontier1 or upflagUpFrontier2 or upflagUpFrontier3 or upflagUpFrontier4)

        if (upflagDownFrontier and flagUpFrontier):
            botFractals[n-fractal_period] = True
            short_entry_third = short_entry_second
            short_entry_second = short_entry_first
            short_entry_first = high[n-fractal_period]
            long_take_profit_value[n] = highest_of_past_length[n] * (1 + long_target_multiple)

            if (not np.isnan(short_entry_first) and not np.isnan(short_entry_second) and not np.isnan(short_entry_third)):
                short_entry_max_value[n] = (max(short_entry_first, short_entry_second, short_entry_third))
                short_entry_min_value[n] = (min(short_entry_first, short_entry_second, short_entry_third))
                short_entry_mid_value[n] = ((short_entry_first + short_entry_second + short_entry_third - short_entry_max_value[n] - short_entry_min_value[n]))
                
                short_entry_max_value[n] = short_entry_max_value[n] * (1 + short_entry_multiple)
                short_entry_min_value[n] = short_entry_min_value[n] * (1 + short_entry_multiple)
                short_entry_mid_value[n] = short_entry_mid_value[n] * (1 + short_entry_multiple)
            
        if (not botFractals[n-fractal_period]): # if botFractals[n] == False, not False = True  
            short_entry_max_value[n] = short_entry_max_value[n-1]
            short_entry_min_value[n] = short_entry_min_value[n-1]
            short_entry_mid_value[n] = short_entry_mid_value[n-1]
            long_take_profit_value[n] = long_take_profit_value[n-1]        
            
        flagDownFrontier = (downflagUpFrontier0 or downflagUpFrontier1 or downflagUpFrontier2 or downflagUpFrontier3 or downflagUpFrontier4)

        if (downflagDownFrontier and flagDownFrontier):
            topFractals[n-fractal_period] = True
            long_entry_third = long_entry_second
            long_entry_second = long_entry_first
            long_entry_first = low[n-fractal_period]
            short_take_profit_value[n] = lowest_of_past_length[n] * (1 - short_target_multiple)

            if (not np.isnan(long_entry_first) and not np.isnan(long_entry_second) and not np.isnan(long_entry_third)):
                long_entry_max_value[n] = (max(long_entry_first, long_entry_second, long_entry_third))
                long_entry_min_value[n] = (min(long_entry_first, long_entry_second, long_entry_third))
                long_entry_mid_value[n] = (long_entry_first + long_entry_second + long_entry_third - long_entry_max_value[n] - long_entry_min_value[n])
                
                long_entry_max_value[n] = long_entry_max_value[n] * (1 - long_entry_multiple)
                long_entry_min_value[n] = long_entry_min_value[n] * (1 - long_entry_multiple)
                long_entry_mid_value[n] = long_entry_mid_value[n] * (1 - long_entry_multiple)

        if (not topFractals[n-fractal_period]): # if topFractals[n] == False, not False = True         
            long_entry_max_value[n] = long_entry_max_value[n-1]
            long_entry_min_value[n] = long_entry_min_value[n-1]
            long_entry_mid_value[n] = long_entry_mid_value[n-1]
            short_take_profit_value[n] = short_take_profit_value[n-1]           

    return (botFractals, 
            topFractals, 
            long_take_profit_value,
            short_entry_max_value,
            short_entry_min_value,
            short_entry_mid_value,
            short_take_profit_value,
            long_entry_max_value,
            long_entry_min_value,
            long_entry_mid_value)
    
######################### Liquidity Purge Strategy - HTF ############################
  
expr = """
LiquidityPurgeStrategy_HTF[lps_HTF]:

# ATR Stop Loss
(short_sl_atr_value,
long_sl_atr_value) = get_sl_atr(@in_HTF_high, 
                                @in_HTF_low, 
                                @in_HTF_close, 
                                @p_sl_period, 
                                @p_sl_multiplier)

# Williams Fractals
highest_of_past_length, lowest_of_past_length = find_highest_lowest(@in_HTF_high, @in_HTF_low, @p_first_target_length_input)

(topFractals, 
botFractals, 
long_take_profit_value, 
short_entry_max_value, 
short_entry_min_value, 
short_entry_mid_value, 
short_take_profit_value, 
long_entry_max_value, 
long_entry_min_value, 
long_entry_mid_value) = will_frac(@in_HTF_high, 
                        @in_HTF_low, 
                        highest_of_past_length, 
                        lowest_of_past_length, 
                        @p_will_frac_period, 
                        @p_short_target_multiple,
                        @p_long_target_multiple,
                        @p_short_entry_multiple,
                        @p_long_entry_multiple)

# Returns
long_sl_atr_value, short_sl_atr_value, \
topFractals, botFractals, \
long_take_profit_value, short_entry_max_value, short_entry_min_value, short_entry_mid_value, \
short_take_profit_value, long_entry_max_value, long_entry_min_value, long_entry_mid_value

"""

LiquidityPurgeStrategy_HTF = vbt.IF.from_expr(
    expr,
    takes_1d=True,

    # ATR Stop Loss
    sl_period=sl_period,
    sl_multiplier=sl_multiplier,
    get_sl_atr=get_sl_atr,

    # Williams Fractals
    first_target_length_input=first_target_length_input,
    will_frac_period=will_frac_period,
    short_target_multiple=short_target_multiple,
    long_target_multiple=long_target_multiple,
    short_entry_multiple=short_entry_multiple,
    long_entry_multiple=long_entry_multiple,
    find_highest_lowest=find_highest_lowest,
    will_frac=will_frac    
)

######################### Liquidity Purge Strategy - LTF ############################
  
expr = """
LiquidityPurgeStrategy_LTF[lps_LTF]:

# ATR Stop Loss
(short_sl_atr_value,
long_sl_atr_value) = get_sl_atr(@in_LTF_high, 
                                @in_LTF_low, 
                                @in_LTF_close, 
                                @p_sl_period, 
                                @p_sl_multiplier)

# Williams Fractals
highest_of_past_length, lowest_of_past_length = find_highest_lowest(@in_LTF_high, @in_LTF_low, @p_first_target_length_input)

(topFractals, 
botFractals, 
long_take_profit_value, 
short_entry_max_value, 
short_entry_min_value, 
short_entry_mid_value, 
short_take_profit_value, 
long_entry_max_value, 
long_entry_min_value, 
long_entry_mid_value) = will_frac(@in_LTF_high, 
                        @in_LTF_low, 
                        highest_of_past_length, 
                        lowest_of_past_length, 
                        @p_will_frac_period, 
                        @p_short_target_multiple,
                        @p_long_target_multiple,
                        @p_short_entry_multiple,
                        @p_long_entry_multiple)

# Returns
long_sl_atr_value, short_sl_atr_value, \
topFractals, botFractals, \
long_take_profit_value, short_entry_max_value, short_entry_min_value, short_entry_mid_value, \
short_take_profit_value, long_entry_max_value, long_entry_min_value, long_entry_mid_value

"""

LiquidityPurgeStrategy_LTF = vbt.IF.from_expr(
    expr,
    takes_1d=True,

    # ATR Stop Loss
    sl_period=sl_period,
    sl_multiplier=sl_multiplier,
    get_sl_atr=get_sl_atr,

    # Williams Fractals
    first_target_length_input=first_target_length_input,
    will_frac_period=will_frac_period,
    short_target_multiple=short_target_multiple,
    long_target_multiple=long_target_multiple,
    short_entry_multiple=short_entry_multiple,
    long_entry_multiple=long_entry_multiple,
    find_highest_lowest=find_highest_lowest,
    will_frac=will_frac    
)

######################### RSI Indicator ##################################

expr = """
RelativeStrengthIndex[rsi]:

# RSI Indicator
rsi_above_upper_threshold, rsi_below_lower_threshold = get_rsi(@in_LTF_close, 
                                                                @p_rsi_period,
                                                                @p_rsi_upper_threshold, 
                                                                @p_rsi_lower_threshold)

# Returns
rsi_above_upper_threshold, rsi_below_lower_threshold

"""

RelativeStrengthIndex = vbt.IF.from_expr(
    expr,
    takes_1d=True,
    
    # RSI Indicator
    rsi_period=rsi_period,
    rsi_upper_threshold=rsi_upper_threshold,
    rsi_lower_threshold=rsi_lower_threshold,
    get_rsi=get_rsi, 
)

####################### Multi-MA Indicator ###############################

expr = """
MultiMA[ma]:

# Value Generation
ma_fast = @talib_sma(@in_HTF_close, @p_ma_fast_period)
ma_mid = @talib_sma(@in_HTF_close, @p_ma_mid_period)
ma_slow = @talib_sma(@in_HTF_close, @p_ma_slow_period)

# Logic for ma_long and ma_short
ma_long = (ma_slow <= ma_mid) & (ma_mid <= ma_fast)
ma_short = (ma_slow >= ma_mid) & (ma_mid >= ma_fast)

# Returns
ma_slow, ma_mid, ma_fast, ma_long, ma_short

"""

MultiMA = vbt.IF.from_expr(
    expr,
    takes_1d=True,
    
    # Multi-MA Indicator
    ma_fast_period=ma_fast_period,
    ma_mid_period=ma_mid_period,
    ma_slow_period=ma_slow_period
)

######################### SuperTrend Indicator ############################
'''
expr = """
SuperTrend[st]:

# SuperTrend Indicator
basic_upper, basic_lower = st_get_basic_bands(@in_HTF_high, @in_HTF_low, @in_HTF_close, @p_st_multiplier, @p_st_period)
trend, dir, long, short = st_get_final_bands(@in_HTF_close, basic_upper, basic_lower)

# Returns
trend, dir, long, short

"""

SuperTrend = vbt.IF.from_expr(
    expr,
    takes_1d=True,

    # SuperTrend Indicator
    st_period=st_period,
    st_multiplier=st_multiplier,
    st_get_basic_bands=st_get_basic_bands,
    st_get_final_bands=st_get_final_bands,
)
'''
#################### Execution Oanda Functions ############################

def check_and_set_flags(api, account_id):
    # Fetch open trades
    trades = api.trade.list_open(account_id).body.get("trades", [])

    # Separate trades into long and short lists
    long_trades = [trade for trade in trades if int(trade.currentUnits) > 0]
    short_trades = [trade for trade in trades if int(trade.currentUnits) < 0]

    # Sort trades by their entry prices
    long_trades = sorted(long_trades, key=lambda x: float(x.price), reverse=True)
    short_trades = sorted(short_trades, key=lambda x: float(x.price))

    l1, l2, l3 = False, False, False
    s1, s2, s3 = False, False, False

    # Set long flags
    if len(long_trades) > 0:
        l1 = True
    if len(long_trades) > 1:
        l2 = True
    if len(long_trades) > 2:
        l3 = True

    # Set short flags
    if len(short_trades) > 0:
        s1 = True
    if len(short_trades) > 1:
        s2 = True
    if len(short_trades) > 2:
        s3 = True

    return l1, l2, l3, s1, s2, s3

def get_account_balance(max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        try:
            account_summary = api.account.get(account_id)
            return float(account_summary.get('account').balance) * 50 # Leverage
        except Exception as e:
            attempts += 1
            time.sleep(1) # Wait for 1 seconds before retrying. You can adjust this as needed.
    raise Exception(f"Failed to get account balance after {max_attempts} attempts.")

def get_instrument_price(instrument):
    try:
        pricing_info = api.pricing.get(account_id, instruments=instrument)
        prices = pricing_info.get('prices')
        return (float(prices[0].bids[0].price) + float(prices[0].asks[0].price)) / 2.0
    except Exception as e:
        return 0.0

def calculate_units(instrument, percent, is_long):
    try:
        # Get account balance
        balance = float(get_account_balance())
        # Calculate amount to invest
        amount = balance * percent
        # Get current instrument price
        price = float(get_instrument_price(instrument))
        # Calculate units
        units = int(amount / price)
        # If the position is short, make units negative
        if not is_long:
            units = -units
        return units
    except Exception as e:
        return 0

def execute_order(instrument, percent, is_long):
        # Calculate units based on account balance and percent
        units = calculate_units(instrument, percent, is_long)
        direction = 'LONG' if is_long else 'SHORT'
        # Execute the order
        response = api.order.market(account_id, instrument=instrument, units=units)

def close_all_positions():
    trades = api.trade.list_open(account_id).body.get("trades", [])
    total_profit_loss = 0

    for trade in trades:
        try:
            # Logging for debugging
            print(f"Processing trade ID: {trade.id}")
            
            # Fetch current trade details
            trade_details = api.trade.get(account_id, trade.id).body["trade"]
            
            # Get the unrealized profit/loss for the current trade
            profit_loss = float(trade_details.unrealizedPL)
            total_profit_loss += profit_loss

            # Close the trade and check response
            response = api.trade.close(account_id, trade.id)
            if not response:  # Or any other check depending on API response
                print(f"Failed to close trade ID: {trade.id}")
            else:
                print(f"Successfully closed trade ID: {trade.id}")
        except Exception as e:
            print(f"Error processing trade ID {trade.id}. Error: {e}")

    # Determine the appropriate message based on total profit/loss
    if total_profit_loss >= 0:
        message = f"Total Profit: ${total_profit_loss:.2f}"
    else:
        message = f"Total Loss: ${-total_profit_loss:.2f}"

    send_pushover_notification(YOUR_USER_KEY, YOUR_API_TOKEN, message)

#################### Streaming Task #####################################

# Shared data structure
latest_msgs = deque(maxlen=1)  # deque with maxlen=1 will only store the latest value

def streaming_task():
    
    r = get_redis_connection(2)
    
    while r.get('stop_streaming') != 'True':
        if stop_event.is_set(): 
            break
        try:        
            try:
                stream = streaming_api.pricing.stream(account_id, snapshot=True, instruments=instrument)
                print("Streaming service set up correctly.")
            except Exception as e:
                print(f"Error setting up streaming service: {str(e)}")

            try:
                test_msg_type, test_msg = next(iter(stream.parts()))
                print(f"Test message type: {test_msg_type}")
                print(f"Test message: {test_msg}")
            except Exception as e:
                print(f"Error retrieving data from stream: {str(e)}")
                
            print("Starting streaming and trading loop...")
    
            while True:
                for msg_type, msg in stream.parts():
                    if msg_type == "pricing.PricingHeartbeat":
                        pass
                    elif msg_type == "pricing.ClientPrice":
                        latest_msgs.append(msg)  # deque will only keep the latest message
                        
        except Exception as e:
            print(f"Error occurred: {str(e)}. Reconnecting...")

        # Add some sleep to avoid rapid-fire reconnection attempts
        time.sleep(5)

def main_loop():
    
    r = get_redis_connection(3)
    
    #################### Starting Values ####################################

    mid_price = np.nan # This will store the latest mid price
    last_mid_price = np.nan # This will store the previous mid price.
    current_long_stop_loss_value = np.nan  # This will store the latest long stop loss value
    current_short_stop_loss_value = np.nan  # This will store the latest short stop loss value
    current_long_take_profit_value = np.nan  # This will store the latest long take profit value
    current_short_take_profit_value = np.nan  # This will store the latest short take profit value
    LTF_granularity = 'M1'
    HTF_granularity = 'M5'
    LTF = fetch_candles(instrument, LTF_granularity, 5000)
    HTF = fetch_candles(instrument, HTF_granularity, 5000)
    last_minute = datetime.now().minute
    
    LTF_open = LTF['Open']
    LTF_high = LTF['High']
    LTF_low = LTF['Low']
    LTF_close = LTF['Close']
    
    HTF_open = HTF['Open']
    HTF_high = HTF['High']
    HTF_low = HTF['Low']
    HTF_close = HTF['Close']

    lps_LTF = LiquidityPurgeStrategy_LTF.run(LTF_high, LTF_low, LTF_close, # calls LTF_high, LTF_low, LTF_close
                                    sl_period=sl_period,
                                    sl_multiplier=sl_multiplier,
                                    first_target_length_input=first_target_length_input,
                                    will_frac_period=will_frac_period,
                                    short_target_multiple=short_target_multiple,
                                    long_target_multiple=long_target_multiple,
                                    short_entry_multiple=short_entry_multiple,
                                    long_entry_multiple=long_entry_multiple,
                                    execute_kwargs=dict(
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))
    '''
    lps_HTF = LiquidityPurgeStrategy_HTF.run(HTF_high, HTF_low, HTF_close, # calls HTF_high, HTF_low, HTF_close
                                    sl_period=sl_period,
                                    sl_multiplier=sl_multiplier,
                                    first_target_length_input=first_target_length_input,
                                    will_frac_period=will_frac_period,
                                    short_target_multiple=short_target_multiple,
                                    long_target_multiple=long_target_multiple,
                                    short_entry_multiple=short_entry_multiple,
                                    long_entry_multiple=long_entry_multiple,
                                    execute_kwargs=dict(
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))
    
    st = SuperTrend.run(HTF_high, HTF_low, HTF_close, # calls HTF_high, HTF_low, HTF_close
                        st_multiplier=st_multiplier,
                        st_period=st_period,                        
                        execute_kwargs=dict(                                        
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))
    '''
    ma = MultiMA.run(HTF_close, # calls HTF_close (5 min candles)
                        ma_fast_period=ma_fast_period,
                        ma_mid_period=ma_mid_period,
                        ma_slow_period=ma_slow_period,                
                        execute_kwargs=dict(                                        
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))    
    
    rsi = RelativeStrengthIndex.run(LTF_close, # calls LTF_close (1 min candles)
                        rsi_period=rsi_period,
                        rsi_upper_threshold=rsi_upper_threshold,
                        rsi_lower_threshold=rsi_lower_threshold,                   
                        execute_kwargs=dict(                                        
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))

    #################### Get Latest Indicator Values #################
    
    #st_dir = st.dir.iloc[-1].item()
    
    ma_long = ma.ma_long.iloc[-1].item()
    ma_short = ma.ma_short.iloc[-1].item()
    ma_fast = ma.ma_fast.iloc[-1].item()
    ma_mid = ma.ma_mid.iloc[-1].item()
    ma_slow = ma.ma_slow.iloc[-1].item()
    
    rsi_above_upper_threshold = rsi.rsi_above_upper_threshold.iloc[-1].item()
    rsi_below_lower_threshold = rsi.rsi_below_lower_threshold.iloc[-1].item()
    
    lps_short_entry_min_value = lps_LTF.short_entry_min_value.iloc[-1].item()
    lps_short_entry_mid_value = lps_LTF.short_entry_mid_value.iloc[-1].item()
    lps_short_entry_max_value = lps_LTF.short_entry_max_value.iloc[-1].item()

    lps_long_entry_min_value = lps_LTF.long_entry_min_value.iloc[-1].item()
    lps_long_entry_mid_value = lps_LTF.long_entry_mid_value.iloc[-1].item()
    lps_long_entry_max_value = lps_LTF.long_entry_max_value.iloc[-1].item()

    lps_short_take_profit_value = lps_LTF.short_take_profit_value.iloc[-1].item()
    lps_long_take_profit_value = lps_LTF.long_take_profit_value.iloc[-1].item()

    lps_short_sl_atr_value = lps_LTF.short_sl_atr_value.iloc[-1].item()
    lps_long_sl_atr_value = lps_LTF.long_sl_atr_value.iloc[-1].item()

    # Initialize a Series with two elements both set to 0
    mid_prices = pd.Series([0, 0])

    # Starting Values
    s1 = False
    s2 = False
    s3 = False  
    l1 = False
    l2 = False
    l3 = False
    s1_check = False
    s2_check = False
    s3_check = False  
    l1_check = False
    l2_check = False
    l3_check = False
    short_stop_1 = 0 # It won't ever be these values - overwritten and sorted up entry.
    short_stop_2 = 0
    short_stop_3 = 0
    long_stop_1 = 1e20 
    long_stop_2 = 1e20
    long_stop_3 = 1e20
    short_tp_1 = 1e20
    short_tp_2 = 1e20
    short_tp_3 = 1e20
    long_tp_1 = 0
    long_tp_2 = 0
    long_tp_3 = 0
    long_tp_hit = False
    long_sl_hit = False
    short_tp_hit = False
    short_sl_hit = False
    long_entry_1_hit = False
    long_entry_2_hit = False
    long_entry_3_hit = False
    short_entry_1_hit = False
    short_entry_2_hit = False
    short_entry_3_hit = False
    spread = False
    #long_cond = False
    #short_cond = True
    
    # Check for open trades
    l1_check, l2_check, l3_check, s1_check, s2_check, s3_check = check_and_set_flags(api, account_id)

    with lock:
        # Update Redis data
        data = decode_data(r)

    for i in range(len(LTF)):
        df = pd.DataFrame({
            'Open time': [LTF.index[i]], # x-axis
            'Open': [LTF['Open'].iloc[i]], # 1-minute candle open
            'High': [LTF['High'].iloc[i]], # 1-minute candle high
            'Low': [LTF['Low'].iloc[i]], # 1-minute candle low
            'Close': [LTF['Close'].iloc[i]], # 1-minute candle close
            # add other data fields here as necessary
        })
        data = pd.concat([data, df], ignore_index=True)
        
    # Convert all Timestamp objects to strings
    data = data.applymap(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f%z") if isinstance(x, pd.Timestamp) else x)

    with lock:
        # Save data back into Redis
        r.set('data', json.dumps(data.to_dict('records')))

    ########################################### Stream Processing Loop ###########################################

    while r.get('stop_streaming') != 'True':                
        if latest_msgs:
            latest_msg = latest_msgs.popleft() # Process latest_msg
                
            ######################## Main Loop Starts #######################

            mid_price = (latest_msg.bids[0].price + latest_msg.asks[0].price) / 2

            mid_prices[0] = last_mid_price
            mid_prices[1] = mid_price
            last_mid_price = mid_price
            
            #################### Get Data For Indicators ####################
            
            # Trim the string to limit the precision to 6 digits
            adjusted_time = adjust_fractional_seconds(latest_msg.time)

            # Convert the adjusted string to a datetime object
            utc_time = datetime.strptime(adjusted_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            utc_time = utc_time.replace(tzinfo=pytz.utc)  # Set its timezone to UTC
            
            # Convert UTC to EST
            est_time = utc_time.astimezone(eastern)

            # Fetch 1-minute data at the start of each new minute
            current_minute = datetime.now().minute
            if current_minute != last_minute:
                print(f"{est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, Fetching New LTF Data...")
                LTF = fetch_candles(instrument, LTF_granularity, 5000)
                LTF_open = LTF['Open']
                LTF_high = LTF['High']
                LTF_low = LTF['Low']
                LTF_close = LTF['Close']
  
            if current_minute != last_minute and current_minute in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
                print(f"{est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, Fetching New HTF Data...")
                HTF = fetch_candles(instrument, HTF_granularity, 5000)
                HTF_open = HTF['Open']
                HTF_high = HTF['High']
                HTF_low = HTF['Low']
                HTF_close = HTF['Close']

            
            lps_LTF = LiquidityPurgeStrategy_LTF.run(LTF_high, LTF_low, LTF_close, # calls LTF_high, LTF_low, LTF_close
                                                sl_period=sl_period,
                                                sl_multiplier=sl_multiplier,
                                                first_target_length_input=first_target_length_input,
                                                will_frac_period=will_frac_period,
                                                short_target_multiple=short_target_multiple,
                                                long_target_multiple=long_target_multiple,
                                                short_entry_multiple=short_entry_multiple,
                                                long_entry_multiple=long_entry_multiple,
                                                execute_kwargs=dict(
                                                    engine='dask',
                                                    chunk_len='auto',
                                                    show_progress=False
                                                    ))
            '''
            lps_HTF = LiquidityPurgeStrategy_HTF.run(HTF_high, HTF_low, HTF_close, # calls HTF_high, HTF_low, HTF_close
                                            sl_period=sl_period,
                                            sl_multiplier=sl_multiplier,
                                            first_target_length_input=first_target_length_input,
                                            will_frac_period=will_frac_period,
                                            short_target_multiple=short_target_multiple,
                                            long_target_multiple=long_target_multiple,
                                            short_entry_multiple=short_entry_multiple,
                                            long_entry_multiple=long_entry_multiple,
                                            execute_kwargs=dict(
                                                engine='dask',
                                                chunk_len='auto',
                                                show_progress=False
                                                ))
            
            st = SuperTrend.run(HTF_high, HTF_low, HTF_close, # calls HTF_high, HTF_low, HTF_close
                                st_multiplier=st_multiplier,
                                st_period=st_period,                        
                                execute_kwargs=dict(                                        
                                                engine='dask',
                                                chunk_len='auto',
                                                show_progress=False
                                                ))
            '''
            ma = MultiMA.run(HTF_close, # calls HTF_close (5 min candles)
                                ma_fast_period=ma_fast_period,
                                ma_mid_period=ma_mid_period,
                                ma_slow_period=ma_slow_period,                
                                execute_kwargs=dict(                                        
                                                engine='dask',
                                                chunk_len='auto',
                                                show_progress=False
                                                ))    
            
            rsi = RelativeStrengthIndex.run(LTF_close, # calls LTF_close (1 min candles)
                                rsi_period=rsi_period,
                                rsi_upper_threshold=rsi_upper_threshold,
                                rsi_lower_threshold=rsi_lower_threshold,                   
                                execute_kwargs=dict(                                        
                                                engine='dask',
                                                chunk_len='auto',
                                                show_progress=False
                                                ))
                
            last_minute = current_minute
                
            #st_dir = st.dir.iloc[-1].item()
            
            ma_long = ma.ma_long.iloc[-1].item()
            ma_short = ma.ma_short.iloc[-1].item()
            ma_fast = ma.ma_fast.iloc[-1].item()
            ma_mid = ma.ma_mid.iloc[-1].item()
            ma_slow = ma.ma_slow.iloc[-1].item()
                    
            rsi_above_upper_threshold = rsi.rsi_above_upper_threshold.iloc[-1].item()
            rsi_below_lower_threshold = rsi.rsi_below_lower_threshold.iloc[-1].item()
            
            lps_short_entry_min_value = lps_LTF.short_entry_min_value.iloc[-1].item()
            lps_short_entry_mid_value = lps_LTF.short_entry_mid_value.iloc[-1].item()
            lps_short_entry_max_value = lps_LTF.short_entry_max_value.iloc[-1].item()

            lps_long_entry_min_value = lps_LTF.long_entry_min_value.iloc[-1].item()
            lps_long_entry_mid_value = lps_LTF.long_entry_mid_value.iloc[-1].item()
            lps_long_entry_max_value = lps_LTF.long_entry_max_value.iloc[-1].item()

            lps_short_take_profit_value = lps_LTF.short_take_profit_value.iloc[-1].item()
            lps_long_take_profit_value = lps_LTF.long_take_profit_value.iloc[-1].item()

            lps_short_sl_atr_value = lps_LTF.short_sl_atr_value.iloc[-1].item()
            lps_long_sl_atr_value = lps_LTF.long_sl_atr_value.iloc[-1].item()      

            ################### Entry/Exit Signals ##########################

            # Short Entries      
            #short_st_cond = st_dir == -1.0
            short_rsi_cond = rsi_above_upper_threshold
            short_ma_cond = ma_short
            short_cond = short_ma_cond & short_rsi_cond
                                                          
            short_entry_1_default_cond = (mid_prices.vbt.crossed_above(lps_short_entry_min_value).iloc[-1].item()) & short_cond
            short_entry_2_default_cond = (mid_prices.vbt.crossed_above(lps_short_entry_mid_value).iloc[-1].item()) & short_cond
            short_entry_3_default_cond = (mid_prices.vbt.crossed_above(lps_short_entry_max_value).iloc[-1].item()) & short_cond

            short_entry_1_no_ma_cond = (mid_prices.vbt.crossed_above(lps_short_entry_min_value).iloc[-1].item()) & short_rsi_cond
            short_entry_2_no_ma_cond = (mid_prices.vbt.crossed_above(lps_short_entry_mid_value).iloc[-1].item()) & short_rsi_cond
            short_entry_3_no_ma_cond = (mid_prices.vbt.crossed_above(lps_short_entry_max_value).iloc[-1].item()) & short_rsi_cond

            # Long Entries
            #long_st_cond = st_dir == 1.0
            long_rsi_cond = rsi_below_lower_threshold
            long_ma_cond = ma_long
            long_cond = long_ma_cond & long_rsi_cond
                                         
            long_entry_1_default_cond = (mid_prices.vbt.crossed_below(lps_long_entry_max_value).iloc[-1].item()) & long_cond
            long_entry_2_default_cond = (mid_prices.vbt.crossed_below(lps_long_entry_mid_value).iloc[-1].item()) & long_cond
            long_entry_3_default_cond = (mid_prices.vbt.crossed_below(lps_long_entry_min_value).iloc[-1].item()) & long_cond

            long_entry_1_no_ma_cond = (mid_prices.vbt.crossed_below(lps_long_entry_max_value).iloc[-1].item()) & long_rsi_cond
            long_entry_2_no_ma_cond = (mid_prices.vbt.crossed_below(lps_long_entry_mid_value).iloc[-1].item()) & long_rsi_cond
            long_entry_3_no_ma_cond = (mid_prices.vbt.crossed_below(lps_long_entry_min_value).iloc[-1].item()) & long_rsi_cond

            ################### Executing Trading Logic ####################

            # Check if the spread is too high
            spread_value = latest_msg.asks[0].price - latest_msg.bids[0].price
            spread = False            
            if spread_value > 0.0002:
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - Spread Too High: ({spread_value})")
                spread = True
            
            if (short_cond):
                msg_streak = "MAs Short"
            elif (long_cond):
                msg_streak = "MAs Long"
            else:
                msg_streak = "MAs Not Aligned"
                
            if (short_cond and (not short_rsi_cond)):
                msg_rsi = "Short RSI Cond Not Met"     
            elif (long_cond and (not long_rsi_cond)):
                msg_rsi = "Long RSI Cond Not Met"
            else:
                msg_rsi = "RSI Cond Met"       
            
            if (l1 or l2 or l3):
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - Long Position Active")
            elif (s1 or s2 or s3):
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - Short Position Active")
            else:
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - {msg_streak} - {msg_rsi}")
            
            ######################## LONGS ############################

            if (l1 or l2 or l3):
                long_entry_1_default_cond = long_entry_1_no_ma_cond
                long_entry_2_default_cond = long_entry_2_no_ma_cond
                long_entry_3_default_cond = long_entry_3_no_ma_cond

            # Long Entries
            if ((long_entry_1_default_cond and \
            #long_cond and \
            (not l1) and \
            (not (s1 or s2 or s3)) and \
            (not np.isnan(lps_long_sl_atr_value)) and \
            (not np.isnan(lps_long_take_profit_value)) and \
            (not spread)) or \
            l1_check):             
                l1 = True # Set Long Entry 1 Position to True                
                # Long Entry 1 Trade Happens Here
                print(f"Long Entry 1 Executed: {latest_msg.instrument} Price: {mid_price}")
                if (not l1_check):
                    execute_order(instrument, 0.33, is_long=True)                
                long_stop_1 = lps_long_sl_atr_value # Set Stop Loss 1 to ATR Stop Loss Value                
                # Set Stop Loss Value to the min of the 3 stop loss values
                current_long_stop_loss_value = min(long_stop_1, min(long_stop_2, long_stop_3))
                long_tp_1 = lps_long_take_profit_value       
                current_long_take_profit_value = max(long_tp_1, max(long_tp_2, long_tp_3))
                long_entry_1_hit = True                
                if (l1_check):
                    l1_check = False
                
            if ((long_entry_2_default_cond and \
            #long_cond and \
            (not l2) and \
            (not (s1 or s2 or s3)) and \
            (not np.isnan(lps_long_sl_atr_value)) and \
            (not np.isnan(lps_long_take_profit_value)) and \
            (not spread)) or \
            l2_check):            
                l2 = True # Set Long Entry 2 Position to True
                # Long Entry 2 Trade Happens Here
                print(f"Long Entry 2 Executed: {latest_msg.instrument} Price: {mid_price}")
                if (not l2_check):
                    execute_order(instrument, 0.33, is_long=True)                
                long_stop_2 = lps_long_sl_atr_value    
                current_long_stop_loss_value = min(long_stop_1, min(long_stop_2, long_stop_3))
                long_tp_2 = lps_long_take_profit_value       
                current_long_take_profit_value = max(long_tp_1, max(long_tp_2, long_tp_3))
                long_entry_2_hit = True                
                if (l2_check):
                    l2_check = False
                
            if ((long_entry_3_default_cond and \
            #long_cond and \
            (not l3) and \
            (not (s1 or s2 or s3)) and \
            (not np.isnan(lps_long_sl_atr_value)) and \
            (not np.isnan(lps_long_take_profit_value)) and \
            (not spread)) or \
            l3_check):            
                l3 = True # Set Long Entry 3 Position to True
                # Long Entry 3 Trade Happens Here
                print(f"Long Entry 3 Executed: {latest_msg.instrument} Price: {mid_price}")
                if (not l3_check):
                    execute_order(instrument, 0.33, is_long=True)                
                long_stop_3 = lps_long_sl_atr_value           
                current_long_stop_loss_value = min(long_stop_1, min(long_stop_2, long_stop_3))
                long_tp_3 = lps_long_take_profit_value       
                current_long_take_profit_value = max(long_tp_1, max(long_tp_2, long_tp_3))
                long_entry_3_hit = True                
                if (l3_check):
                    l3_check = False

            # Long Take Profit Exits
            if (l1 or l2 or l3) and (mid_prices.vbt.crossed_above(current_long_take_profit_value).iloc[-1].item()):                
                # Long Close Trade Happens Here
                print(f"Closing All Long Positions (Take Profit): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()
                l1 = False
                l2 = False
                l3 = False
                long_tp_hit = True
                #short_cond = False
                #long_cond = True

            # Long Stop Loss Exit    
            if (l1 or l2 or l3) and (mid_prices.vbt.crossed_below(current_long_stop_loss_value).iloc[-1].item()):
                # Long Close Trade Happens Here
                print(f"Closing All Long Positions (Stop-Loss): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()                   
                l1 = False
                l2 = False
                l3 = False
                long_sl_hit = True
                #short_cond = True
                #long_cond = False
                
            ######################## SHORTS ############################

            if (s1 or s2 or s3):
                short_entry_1_default_cond = short_entry_1_no_ma_cond
                short_entry_2_default_cond = short_entry_2_no_ma_cond
                short_entry_3_default_cond = short_entry_3_no_ma_cond

            # Short Entries
            if ((short_entry_1_default_cond and \
            #short_cond and \
            (not s1) and \
            (not (l1 or l2 or l3)) and \
            (not np.isnan(lps_short_sl_atr_value)) and \
            (not np.isnan(lps_short_take_profit_value)) and \
            (not spread)) or \
            s1_check):           
                s1 = True # Set Entry 1 Position to True
                # Short Entry 1 Trade Happens Here
                print(f"Short Entry 1 Executed: {latest_msg.instrument} Price: {mid_price}")
                if (not s1_check):
                    execute_order(instrument, 0.33, is_long=False)                
                short_stop_1 = lps_short_sl_atr_value # Set Stop Loss 1 to ATR Stop Loss Value                
                # Set Stop Loss Value to the max of the 3 stop loss values
                current_short_stop_loss_value = max(short_stop_1, max(short_stop_2, short_stop_3))
                short_tp_1 = lps_short_take_profit_value       
                current_short_take_profit_value = min(short_tp_1, min(short_tp_2, short_tp_3))
                short_entry_1_hit = True                
                if (s1_check):
                    s1_check = False
                
            if ((short_entry_2_default_cond and \
            #short_cond and \
            (not s2) and \
            (not (l1 or l2 or l3)) and \
            (not np.isnan(lps_short_sl_atr_value)) and \
            (not np.isnan(lps_short_take_profit_value)) and \
            (not spread)) or \
            s2_check):            
                s2 = True
                # Short Entry 2 Trade Happens Here
                print(f"Short Entry 2 Executed: {latest_msg.instrument} Price: {mid_price}")
                if (not s2_check):
                    execute_order(instrument, 0.33, is_long=False)                
                short_stop_2 = lps_short_sl_atr_value       
                current_short_stop_loss_value = max(short_stop_1, max(short_stop_2, short_stop_3))
                short_tp_2 = lps_short_take_profit_value       
                current_short_take_profit_value = min(short_tp_1, min(short_tp_2, short_tp_3))
                short_entry_2_hit = True                
                if (s2_check):
                    s2_check = False
                
            if ((short_entry_3_default_cond and \
            #short_cond and \
            (not s3) and \
            (not (l1 or l2 or l3)) and \
            (not np.isnan(lps_short_sl_atr_value)) and \
            (not np.isnan(lps_short_take_profit_value)) and \
            (not spread)) or \
            s3_check):             
                s3 = True
                # Short Entry 3 Trade Happens Here
                print(f"Short Entry 3 Executed: {latest_msg.instrument} Price: {mid_price}")
                if (not s3_check):
                    execute_order(instrument, 0.33, is_long=False)                
                short_stop_3 = lps_short_sl_atr_value       
                current_short_stop_loss_value = max(short_stop_1, max(short_stop_2, short_stop_3))
                short_tp_3 = lps_short_take_profit_value       
                current_short_take_profit_value = min(short_tp_1, min(short_tp_2, short_tp_3))
                short_entry_3_hit = True                
                if (s3_check):
                    s3_check = False
                
            # Short Take Profit Exit
            if (s1 or s2 or s3) and (mid_prices.vbt.crossed_below(current_short_take_profit_value).iloc[-1].item()):                
                # Short Close Trade Happens Here
                print(f"Closing All Short Positions (Take Profit): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()
                s1 = False
                s2 = False
                s3 = False
                short_tp_hit = True
                #short_cond = True
                #long_cond = False

            # Short Stop Loss Exit
            if (s1 or s2 or s3) and (mid_prices.vbt.crossed_above(current_short_stop_loss_value).iloc[-1].item()):                    
                # Short Close Trade Happens Here
                print(f"Closing All Short Positions (Stop-Loss): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()           
                s1 = False
                s2 = False
                s3 = False
                short_sl_hit = True
                #short_cond = False
                #long_cond = True

            # Update Dash App Chart:
            with lock:
                # Update Redis data
                data = decode_data(r)
                
            df = pd.DataFrame({
                'Open time': [LTF.index[-1]], # x-axis
                'Open': [LTF_open.iloc[-1].item()], # 1-minute candle open
                'High': [LTF_high.iloc[-1].item()], # 1-minute candle high
                'Low': [LTF_low.iloc[-1].item()], # 1-minute candle low
                'Close': [LTF_close.iloc[-1].item()], # 1-minute candle close
                'short cond': [short_cond], # Short condition
                'long cond' : [long_cond], # Long condition
                'ma fast' : [ma_fast], # fast moving average
                'ma mid' : [ma_mid], # mid moving average
                'ma slow' : [ma_slow], # slow moving average
                's1' : [s1], # short entry 1 active
                's2' : [s2], # short entry 2 active
                's3' : [s3], # short entry 3 active
                'l1' : [l1], # long entry 1 active
                'l2' : [l2], # long entry 2 active
                'l3' : [l3], # long entry 3 active
                # short entry 1 scatter plot - True/False on whether date index should have a mark. (white)
                'short entry 1': [short_entry_1_hit],
                # short entry 2 scatter plot - True/False on whether date index should have a mark. (yellow)
                'short entry 2': [short_entry_2_hit], 
                # short entry 3 scatter plot - True/False on whether date index should have a mark. (green)
                'short entry 3': [short_entry_3_hit], 
                # same as above but for long entries.
                'long entry 1': [long_entry_1_hit],
                'long entry 2': [long_entry_2_hit],
                'long entry 3': [long_entry_3_hit],
                # short stop-loss hit scatter plot - True/False on whether date index should have a mark. (red)
                'short stop-loss hit': [short_sl_hit], 
                'long stop-loss hit': [long_sl_hit], # same as above, long stop-loss hit scatter plot (red)
                'short take-profit hit': [short_tp_hit], # same as above, short take-profit hit scatter plot (lightblue)
                'long take-profit hit': [long_tp_hit], # same as above, long take-profit hit scatter plot (lightblue)
                'bids': [latest_msg.bids[0].price], # bid price
                'asks': [latest_msg.asks[0].price], # ask price
                # line plot (white) and y-axis level for short entry 1 - only show when 'st dir' is -1
                'short entry 1 level': [lps_short_entry_min_value], 
                # line plot (yellow) and y-axis level for short entry 2 - only show when 'st dir' is -1
                'short entry 2 level': [lps_short_entry_mid_value],
                # line plot (green) and y-axis level for short entry 3 - only show when 'st dir' is -1
                'short entry 3 level': [lps_short_entry_max_value],
                # line plot (white) and y-axis level for long entry 1 - only show when 'st dir' is 1
                'long entry 1 level': [lps_long_entry_max_value],
                # line plot (yellow) and y-axis level for long entry 2 - only show when 'st dir' is 1
                'long entry 2 level': [lps_long_entry_mid_value],
                # line plot (green) and y-axis level for long entry 3 - only show when 'st dir' is 1
                'long entry 3 level': [lps_long_entry_min_value],
                # line plot (red) and y-axis level for short stop-loss hit - only show (s1 or s2 or s3) == True
                'short stop-loss': [current_short_stop_loss_value],
                # line plot (red) and y-axis level for long stop-loss hit - only show (l1 or l2 or l3) == True
                'long stop-loss': [current_long_stop_loss_value],
                # line plot (lightblue) and y-axis level for short take-profit hit - only show (s1 or s2 or s3) == True
                'short take-profit': [current_short_take_profit_value],
                # line plot (lightblue) and y-axis level for long take-profit hit - only show (l1 or l2 or l3) == True
                'long take-profit': [current_long_take_profit_value],
            })
            
            data = pd.concat([data, df], ignore_index=True)
                
            # Convert all Timestamp objects to strings
            data = data.applymap(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f%z") if isinstance(x, pd.Timestamp) else x)

            with lock:
                # Save data back into Redis
                r.set('data', json.dumps(data.to_dict('records')))

            # Revert the plotting triggers to False
            long_tp_hit = False
            long_sl_hit = False
            short_tp_hit = False
            short_sl_hit = False
            long_entry_1_hit = False
            long_entry_2_hit = False
            long_entry_3_hit = False
            short_entry_1_hit = False
            short_entry_2_hit = False
            short_entry_3_hit = False
            
            # Add a sleep interval to avoid consuming too much CPU
            time.sleep(0.1)  # Sleep for 100ms  
            
# Create and start the streaming thread
stream_thread = threading.Thread(target=streaming_task)
stream_thread.start()  
        
# Start the main_loop thread
main_loop = threading.Thread(target=main_loop)
main_loop.start()
            
if __name__ == '__main__':    
    app.run_server(host='0.0.0.0', port=8050, debug=True, use_reloader=False) # Run the Dash app